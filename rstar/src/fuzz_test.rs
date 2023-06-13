#[cfg(test)]
mod test {
    use crate::point::PointExt;
    use crate::primitives::Line;
    use crate::{
        Envelope, ParentNode, Point, PointDistance, RTree, RTreeNode, RTreeObject,
        SelectionFunction,
    };
    use rand::distributions::Uniform;
    use rand::{Rng, SeedableRng};
    use rand_hc::Hc128Rng;
    use smallvec::SmallVec;
    use std::collections::HashSet;

    const ENVELOPE_TOLERANCE: f32 = 1e-2;
    const POINT_SIMILARITY_THRESHOLD: f32 = 1e-2;

    #[derive(Copy, Clone)]
    struct FuzzTestLine {
        id: usize,
        line: Line<[f32; 3]>,
    }

    impl RTreeObject for FuzzTestLine {
        type Envelope = crate::AABB<[f32; 3]>;

        fn envelope(&self) -> Self::Envelope {
            self.line.envelope()
        }
    }

    impl PointDistance for FuzzTestLine {
        fn distance_2(
            &self,
            point: &<Self::Envelope as Envelope>::Point,
        ) -> <<Self::Envelope as Envelope>::Point as Point>::Scalar {
            self.line.distance_2(point)
        }
    }

    struct DrainById(FuzzTestLine);

    impl SelectionFunction<FuzzTestLine> for DrainById {
        fn should_unpack_parent(&self, envelope: &crate::AABB<[f32; 3]>) -> bool {
            let envelope = envelope.clone();
            let mut lower = envelope.lower();
            let mut upper = envelope.upper();
            lower[0] -= ENVELOPE_TOLERANCE;
            lower[1] -= ENVELOPE_TOLERANCE;
            lower[2] -= ENVELOPE_TOLERANCE;
            upper[0] += ENVELOPE_TOLERANCE;
            upper[1] += ENVELOPE_TOLERANCE;
            upper[2] += ENVELOPE_TOLERANCE;
            let upper = envelope.upper();
            let envelope = crate::AABB::from_corners(lower, upper);
            envelope.intersects(&self.0.line.envelope())
        }

        fn should_unpack_leaf(&self, leaf: &FuzzTestLine) -> bool {
            leaf.id == self.0.id
        }
    }

    struct FuzzTesterDistributions {
        pub range: Uniform<f32>,
        pub max_line_insertions_in_one_batch: usize,
        pub max_line_removals_in_one_batch: usize,
        pub remove_added_line_probability: f64,
        pub remove_added_line_local_probability: f64,
        pub remove_similar_lines_probability: f64,
        pub remove_similar_lines_local_probability: f64,
        pub generate_line_from_existing_probability: f64,
    }

    struct FuzzTesterConfig {
        pub rng_seed: [u8; 32],
        pub distributions: FuzzTesterDistributions,
    }

    struct FuzzTester {
        pub tree: RTree<FuzzTestLine>,
        pub map: std::collections::HashMap<usize, FuzzTestLine>,
        pub set: std::collections::BTreeSet<usize>,
        pub current_id: usize,
        pub rng: Hc128Rng,
        pub distributions: FuzzTesterDistributions,
    }


    impl FuzzTester {
        pub fn new(config: FuzzTesterConfig) -> FuzzTester {
            Self {
                tree: Default::default(),
                map: Default::default(),
                set: Default::default(),
                current_id: 0,
                rng: Hc128Rng::from_seed(config.rng_seed),
                distributions: config.distributions,
            }
        }
        pub fn bulk_load_self(&mut self) {
            self.tree = RTree::bulk_load(self.tree.iter().map(|segment| segment.clone()).collect());
        }

        fn generate_random_point(&mut self) -> [f32; 3] {
            [
                self.rng.sample(self.distributions.range),
                self.rng.sample(self.distributions.range),
                self.rng.sample(self.distributions.range),
            ]
        }

        fn get_random_existing_line(&mut self) -> Line<[f32; 3]> {
            // Try to find a random line by finding the nearest 5 lines to the probe point
            let p = self.generate_random_point();
            let candidates = self
                .tree
                .nearest_neighbor_iter(&p)
                .take(5)
                .cloned()
                .collect::<SmallVec<[FuzzTestLine; 5]>>();
            if candidates.len() > 0 {
                // Pick a random line from the 5 neaest lines
                candidates[self.rng.gen_range(0, candidates.len())].line
            } else {
                // Fallback to a random line
                Line::new(self.generate_random_point(), self.generate_random_point())
            }
        }

        fn generate_line_from_existing_line_or_point(&mut self) -> Line<[f32; 3]> {
            match self.rng.gen_range(0, 3) {
                // Duplicate line
                0 => self.get_random_existing_line(),
                // Connect two random lines
                1 => {
                    let line1 = self.get_random_existing_line();
                    let line2 = self.get_random_existing_line();
                    Line::new(
                        [line1.from, line1.to][self.rng.gen_range(0, 2)],
                        [line2.from, line2.to][self.rng.gen_range(0, 2)],
                    )
                }
                // Connect random line to new point
                2 => {
                    let line = self.get_random_existing_line();
                    let random_point = self.generate_random_point();
                    let from_or_to = self.rng.gen_range(0, 2);
                    self.shuffle_line_points(Line::new(
                        [line.from, line.to][from_or_to],
                        random_point,
                    ))
                }
                _ => unreachable!(),
            }
        }

        fn shuffle_line_points(&mut self, line: Line<[f32; 3]>) -> Line<[f32; 3]> {
            if self.rng.gen_bool(0.5) {
                Line::new(line.to, line.from)
            } else {
                line
            }
        }

        fn generate_line(&mut self) -> Line<[f32; 3]> {
            if self.rng.gen_bool(self.distributions.generate_line_from_existing_probability) {
                self.generate_line_from_existing_line_or_point()
            } else {
                Line::new(self.generate_random_point(), self.generate_random_point())
            }
        }

        fn run_test(&mut self, ops_count: u64) {
            for _ in 0..ops_count {
                match self.rng.gen_range(0, 2) {
                    0 => {
                        self.add_items();
                    }
                    1 => {
                        self.remove_items();
                    }
                    _ => unreachable!(),
                }
            }
        }

        fn add_items(&mut self) {
            let to_insert = self
                .rng
                .gen_range(1, self.distributions.max_line_insertions_in_one_batch);
            let remove_newly_inserted_line = self
                .rng
                .gen_bool(self.distributions.remove_added_line_probability);
            for _ in 0..to_insert {
                let line = self.generate_line();
                let repeats = if self.rng.gen_bool(0.2) {
                    self.rng.gen_range(1, 10)
                } else {
                    1
                };
                for _ in 0..repeats {
                    let inserted = self.insert_line(line);
                    // Remove the newly inserted line
                    if remove_newly_inserted_line
                        && self
                        .rng
                        .gen_bool(self.distributions.remove_added_line_local_probability)
                    {
                        self.remove_line_by_id(&inserted.id);
                    }
                }
            }
        }

        fn remove_items(&mut self) {
            let n_to_remove = self
                .rng
                .gen_range(1, self.distributions.max_line_removals_in_one_batch);
            let remove_similar_lines = self
                .rng
                .gen_bool(self.distributions.remove_similar_lines_probability);
            for _ in 0..n_to_remove {
                if self.map.len() == 0 {
                    break;
                }
                let to_remove_id = self.random_segment_id();
                let removed = self.remove_line_by_id(&to_remove_id);
                // Remove all similar lines
                if remove_similar_lines
                    && self
                    .rng
                    .gen_bool(self.distributions.remove_similar_lines_local_probability)
                {
                    self.remove_similar_lines(&removed);
                }
            }
        }

        fn remove_similar_lines(&mut self, line: &FuzzTestLine) {
            let ids = self
                .tree
                .locate_in_envelope_intersecting(&line.line.envelope())
                .filter(|x| {
                    (PointExt::distance_2(&x.line.from, &line.line.from) < POINT_SIMILARITY_THRESHOLD
                        && PointExt::distance_2(&x.line.to, &line.line.to) < POINT_SIMILARITY_THRESHOLD)
                        || (PointExt::distance_2(&x.line.from, &line.line.to) < POINT_SIMILARITY_THRESHOLD
                        && PointExt::distance_2(&x.line.to, &line.line.from) < POINT_SIMILARITY_THRESHOLD)
                })
                .map(|matching_line| matching_line.id)
                .collect::<HashSet<_>>();
            for id in ids {
                self.remove_line_by_id(&id);
            }
        }

        fn random_segment_id(&mut self) -> usize {
            let k = self.rng.gen_range(0, self.set.len());
            let to_remove_id = self
                .set
                .range(k..)
                .next()
                .cloned()
                .expect("No longer in collection");
            to_remove_id
        }

        fn remove_line_by_id(&mut self, to_remove_id: &usize) -> FuzzTestLine {
            let to_remove = *self.map.get(&to_remove_id).unwrap();
            self.tree
                .remove_with_selection_function(DrainById(to_remove));
            self.map.remove(&to_remove_id);
            self.set.remove(&to_remove_id);
            return to_remove;
        }

        fn insert_line(&mut self, line: Line<[f32; 3]>) -> FuzzTestLine {
            let id = self.current_id;
            let segment = FuzzTestLine { id: id, line: line };
            self.tree.insert(segment);
            self.map.insert(id, segment);
            self.set.insert(id);
            self.current_id += 1;
            segment.clone()
        }
    }

    #[test]
    fn test_fuzz_from_empty() {
        let mut tester = FuzzTester::new(FuzzTesterConfig {
            rng_seed: *b"QFoXFrpYToCe1B71wPYxAkIiHcEmSBAx",
            distributions: FuzzTesterDistributions {
                range: Uniform::from(-30.0_f32..30.0_f32),
                max_line_insertions_in_one_batch: 2,
                max_line_removals_in_one_batch: 2,
                remove_added_line_probability: 0.1,
                remove_added_line_local_probability: 0.1,
                remove_similar_lines_probability: 0.1,
                remove_similar_lines_local_probability: 0.1,
                generate_line_from_existing_probability: 0.2,
            },
        });
        tester.run_test(10000);
    }

    #[test]
    fn test_fuzz_from_bulk() {
        let mut tester = FuzzTester::new(FuzzTesterConfig {
            rng_seed: *b"QFoXFrpYToCe1B71wPYxAkIiHcEmSBAx",
            distributions: FuzzTesterDistributions {
                range: Uniform::from(-30.0_f32..30.0_f32),
                max_line_insertions_in_one_batch: 2,
                max_line_removals_in_one_batch: 2,
                remove_added_line_probability: 0.1,
                remove_added_line_local_probability: 0.1,
                remove_similar_lines_probability: 0.1,
                remove_similar_lines_local_probability: 0.1,
                generate_line_from_existing_probability: 0.2,
            },
        });
        while tester.tree.size() < 10000 {
            tester.add_items();
        }
        tester.bulk_load_self();
        tester.run_test(10000);
    }
}
