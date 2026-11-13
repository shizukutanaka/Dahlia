//! Identity domain logic (clustering evaluation, quality scores).
//! Filled in Phase 8; the storage side lives in core/storage.

use std::collections::HashSet;

/// Mean pairwise cosine similarity of a set of unit vectors is the
/// identity quality signal; implemented with the clustering phase.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// A previously stored identity, as a candidate for name/ID carry-over
/// when clustering runs again.
#[derive(Debug, Clone)]
pub struct PreviousIdentity {
    pub id: i64,
    pub name: String,
    pub face_ids: Vec<i64>,
}

/// Minimum overlap coefficient (shared faces / smaller set) for a new
/// cluster to be considered "the same person" as a previous identity.
pub const CARRY_OVER_MIN_OVERLAP: f64 = 0.5;

/// Decide which previous identity each new cluster continues.
///
/// Face IDs are stable across clustering runs (clustering never touches
/// the faces table; each face carries its embedding), so the face-set
/// overlap identifies the same person even when cluster boundaries move
/// a little. Matching is greedy over candidate pairs sorted by overlap
/// coefficient (shared / min(|prev|, |new|)), then by shared count;
/// each previous identity and each new cluster is used at most once.
///
/// Returns one entry per group in `groups`: the index into `previous`
/// of the identity it continues, or None for a genuinely new person.
pub fn match_previous(
    previous: &[PreviousIdentity],
    groups: &[Vec<i64>],
) -> Vec<Option<usize>> {
    let prev_sets: Vec<HashSet<i64>> =
        previous.iter().map(|p| p.face_ids.iter().copied().collect()).collect();
    let group_sets: Vec<HashSet<i64>> =
        groups.iter().map(|g| g.iter().copied().collect()).collect();

    let mut candidates: Vec<(f64, usize, usize, usize)> = Vec::new(); // (coef, shared, prev, group)
    for (pi, ps) in prev_sets.iter().enumerate() {
        if ps.is_empty() {
            continue;
        }
        for (gi, gs) in group_sets.iter().enumerate() {
            if gs.is_empty() {
                continue;
            }
            let shared = ps.intersection(gs).count();
            if shared == 0 {
                continue;
            }
            let coef = shared as f64 / ps.len().min(gs.len()) as f64;
            if coef >= CARRY_OVER_MIN_OVERLAP {
                candidates.push((coef, shared, pi, gi));
            }
        }
    }
    // Best overlap first; deterministic tie-break on indices.
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });

    let mut result: Vec<Option<usize>> = vec![None; groups.len()];
    let mut used_prev: HashSet<usize> = HashSet::new();
    let mut used_group: HashSet<usize> = HashSet::new();
    for (_, _, pi, gi) in candidates {
        if used_prev.contains(&pi) || used_group.contains(&gi) {
            continue;
        }
        used_prev.insert(pi);
        used_group.insert(gi);
        result[gi] = Some(pi);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prev(id: i64, faces: &[i64]) -> PreviousIdentity {
        PreviousIdentity { id, name: format!("P{id}"), face_ids: faces.to_vec() }
    }

    #[test]
    fn identical_groups_carry_over() {
        let previous = vec![prev(1, &[1, 2, 3]), prev(2, &[4, 5])];
        let groups = vec![vec![4, 5], vec![1, 2, 3]];
        assert_eq!(match_previous(&previous, &groups), vec![Some(1), Some(0)]);
    }

    #[test]
    fn mostly_same_group_carries_over_despite_small_drift() {
        // One face moved out, one moved in: 3 of 4 shared.
        let previous = vec![prev(1, &[1, 2, 3, 4])];
        let groups = vec![vec![1, 2, 3, 9]];
        assert_eq!(match_previous(&previous, &groups), vec![Some(0)]);
    }

    #[test]
    fn split_group_follows_the_larger_overlap() {
        // A previous identity split in two: only the part that keeps at
        // least half of the smaller set may continue the identity.
        let previous = vec![prev(1, &[1, 2, 3, 4, 5, 6])];
        let groups = vec![vec![1, 2], vec![3, 4, 5, 6]];
        let m = match_previous(&previous, &groups);
        assert_eq!(m[1], Some(0));
        assert_eq!(m[0], None);
    }

    #[test]
    fn unrelated_groups_do_not_match() {
        let previous = vec![prev(1, &[1, 2, 3])];
        let groups = vec![vec![7, 8, 9]];
        assert_eq!(match_previous(&previous, &groups), vec![None]);
    }

    #[test]
    fn one_previous_identity_matches_at_most_one_group() {
        let previous = vec![prev(1, &[1, 2, 3, 4])];
        let groups = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4]];
        let m = match_previous(&previous, &groups);
        assert_eq!(m.iter().filter(|x| x.is_some()).count(), 1);
    }

    #[test]
    fn empty_previous_faces_never_match() {
        let previous = vec![prev(1, &[])];
        let groups = vec![vec![1, 2]];
        assert_eq!(match_previous(&previous, &groups), vec![None]);
    }
}
