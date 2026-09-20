//! Dense calendar-aligned numerical primitives matching the Python/Polars oracle.

pub type Series = Vec<f64>;

#[derive(Clone, Debug)]
pub struct Shape {
    pub dates: usize,
    pub codes: usize,
}

impl Shape {
    pub fn len(&self) -> usize {
        self.dates * self.codes
    }
    #[inline]
    pub fn at(&self, date: usize, code: usize) -> usize {
        date * self.codes + code
    }
}

pub fn unary(x: &[f64], f: impl Fn(f64) -> f64) -> Series {
    x.iter()
        .map(|v| if v.is_finite() { f(*v) } else { f64::NAN })
        .collect()
}
pub fn binary(a: &[f64], b: &[f64], f: impl Fn(f64, f64) -> f64) -> Series {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x.is_finite() && y.is_finite() {
                f(*x, *y)
            } else {
                f64::NAN
            }
        })
        .collect()
}
pub fn scalar(a: &[f64], b: f64, f: impl Fn(f64, f64) -> f64) -> Series {
    a.iter()
        .map(|x| if x.is_finite() { f(*x, b) } else { f64::NAN })
        .collect()
}
pub fn choose(condition: &[f64], yes: &[f64], no: &[f64]) -> Series {
    condition
        .iter()
        .zip(yes)
        .zip(no)
        .map(|((c, y), n)| {
            // Polars `when(...).then(...).otherwise(...)` treats a null
            // predicate as false.  The Python oracle relies on that behavior
            // for the first lagged row and for calendar gaps.
            if c.is_finite() && *c != 0.0 { *y } else { *n }
        })
        .collect()
}
pub fn delay(shape: &Shape, x: &[f64], periods: usize) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    for d in periods..shape.dates {
        for c in 0..shape.codes {
            out[shape.at(d, c)] = x[shape.at(d - periods, c)];
        }
    }
    out
}
pub fn delta(shape: &Shape, x: &[f64], periods: usize) -> Series {
    binary(x, &delay(shape, x, periods), |a, b| a - b)
}

#[derive(Clone, Copy)]
pub enum Rolling {
    Sum,
    Mean,
    Std,
    Min,
    Max,
}
pub fn rolling(shape: &Shape, x: &[f64], window: usize, kind: Rolling) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    if window == 0 {
        return out;
    }
    for c in 0..shape.codes {
        for d in window - 1..shape.dates {
            let values = (d + 1 - window..=d)
                .map(|k| x[shape.at(k, c)])
                .collect::<Vec<_>>();
            if values.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let value = match kind {
                Rolling::Sum => values.iter().sum(),
                Rolling::Mean => values.iter().sum::<f64>() / window as f64,
                Rolling::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
                Rolling::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                Rolling::Std => {
                    if window < 2 {
                        f64::NAN
                    } else {
                        let m = values.iter().sum::<f64>() / window as f64;
                        (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (window - 1) as f64)
                            .sqrt()
                    }
                }
            };
            out[shape.at(d, c)] = value;
        }
    }
    out
}
pub fn corr(shape: &Shape, a: &[f64], b: &[f64], window: usize) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    if window < 2 {
        return out;
    }
    for c in 0..shape.codes {
        for d in window - 1..shape.dates {
            let mut sa = 0.0;
            let mut sb = 0.0;
            let mut ok = true;
            for k in d + 1 - window..=d {
                let x = a[shape.at(k, c)];
                let y = b[shape.at(k, c)];
                if !x.is_finite() || !y.is_finite() {
                    ok = false;
                    break;
                }
                sa += x;
                sb += y
            }
            if !ok {
                continue;
            }
            let ma = sa / window as f64;
            let mb = sb / window as f64;
            let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
            for k in d + 1 - window..=d {
                let x = a[shape.at(k, c)] - ma;
                let y = b[shape.at(k, c)] - mb;
                cov += x * y;
                va += x * x;
                vb += y * y
            }
            let denom = (va * vb).sqrt();
            if denom > 0.0 {
                out[shape.at(d, c)] = cov / denom
            }
        }
    }
    out
}
pub fn covariance(shape: &Shape, a: &[f64], b: &[f64], window: usize) -> Series {
    let correlation = corr(shape, a, b, window);
    let sa = rolling(shape, a, window, Rolling::Std);
    let sb = rolling(shape, b, window, Rolling::Std);
    binary(&correlation, &binary(&sa, &sb, |x, y| x * y), |r, s| r * s)
}
/// Minimum rank, matching Polars/Pandas; ties receive the first rank.
fn min_rank(values: &[(usize, f64)]) -> Vec<(usize, f64)> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut out = Vec::with_capacity(sorted.len());
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j].1 == sorted[i].1 {
            j += 1
        }
        for item in &sorted[i..j] {
            out.push((item.0, (i + 1) as f64))
        }
        i = j
    }
    out
}
pub fn cross_rank(shape: &Shape, x: &[f64]) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    for d in 0..shape.dates {
        let values = (0..shape.codes)
            .filter_map(|c| {
                let v = x[shape.at(d, c)];
                v.is_finite().then_some((c, v))
            })
            .collect::<Vec<_>>();
        let n = values.len() as f64;
        for (c, r) in min_rank(&values) {
            out[shape.at(d, c)] = r / n
        }
    }
    out
}
pub fn cross_extreme(shape: &Shape, x: &[f64], maximum: bool) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    for d in 0..shape.dates {
        let value = (0..shape.codes)
            .filter_map(|c| {
                let v = x[shape.at(d, c)];
                v.is_finite().then_some(v)
            })
            .reduce(|a, b| if maximum { a.max(b) } else { a.min(b) });
        if let Some(value) = value {
            for c in 0..shape.codes {
                out[shape.at(d, c)] = value;
            }
        }
    }
    out
}
pub fn rolling_beta_skip_nulls(
    shape: &Shape,
    a: &[f64],
    b: &[f64],
    window: usize,
    min_samples: usize,
) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    for c in 0..shape.codes {
        for d in 0..shape.dates {
            let begin = (d + 1).saturating_sub(window);
            let pairs = (begin..=d)
                .filter_map(|k| {
                    let x = a[shape.at(k, c)];
                    let y = b[shape.at(k, c)];
                    (x.is_finite() && y.is_finite()).then_some((x, y))
                })
                .collect::<Vec<_>>();
            if pairs.len() < min_samples {
                continue;
            }
            let n = pairs.len() as f64;
            let ma = pairs.iter().map(|x| x.0).sum::<f64>() / n;
            let mb = pairs.iter().map(|x| x.1).sum::<f64>() / n;
            let cov = pairs.iter().map(|x| (x.0 - ma) * (x.1 - mb)).sum::<f64>() / (n - 1.0);
            let var = pairs.iter().map(|x| (x.1 - mb).powi(2)).sum::<f64>() / (n - 1.0);
            if var > 0.0 {
                out[shape.at(d, c)] = cov / var;
            }
        }
    }
    out
}
pub fn ts_rank(shape: &Shape, x: &[f64], window: usize) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    if window == 0 {
        return out;
    }
    for c in 0..shape.codes {
        for d in window - 1..shape.dates {
            let values = (d + 1 - window..=d)
                .map(|k| (k, x[shape.at(k, c)]))
                .collect::<Vec<_>>();
            if values.iter().any(|v| !v.1.is_finite()) {
                continue;
            }
            for (i, r) in min_rank(&values) {
                if i == d {
                    out[shape.at(d, c)] = r;
                    break;
                }
            }
        }
    }
    out
}
pub fn ts_argmax(shape: &Shape, x: &[f64], window: usize) -> Series {
    ts_arg(shape, x, window, true)
}
pub fn ts_argmin(shape: &Shape, x: &[f64], window: usize) -> Series {
    ts_arg(shape, x, window, false)
}
fn ts_arg(shape: &Shape, x: &[f64], window: usize, maximize: bool) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    if window == 0 {
        return out;
    }
    for c in 0..shape.codes {
        for d in window - 1..shape.dates {
            let mut best = None;
            for (pos, k) in (d + 1 - window..=d).enumerate() {
                let v = x[shape.at(k, c)];
                if !v.is_finite() {
                    best = None;
                    break;
                }
                if best
                    .map(|(_, b)| if maximize { v > b } else { v < b })
                    .unwrap_or(true)
                {
                    best = Some((pos, v))
                }
            }
            if let Some((pos, _)) = best {
                out[shape.at(d, c)] = (pos + 1) as f64
            }
        }
    }
    out
}
pub fn decay_linear(shape: &Shape, x: &[f64], window: usize) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    let divisor = (window * (window + 1) / 2) as f64;
    if window == 0 {
        return out;
    }
    for c in 0..shape.codes {
        for d in window - 1..shape.dates {
            let mut sum = 0.0;
            let mut ok = true;
            for (pos, k) in (d + 1 - window..=d).enumerate() {
                let v = x[shape.at(k, c)];
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                sum += v * (pos + 1) as f64
            }
            if ok {
                out[shape.at(d, c)] = sum / divisor
            }
        }
    }
    out
}
pub fn ewm(shape: &Shape, x: &[f64], alpha: f64) -> Series {
    let mut out = vec![f64::NAN; shape.len()];
    for c in 0..shape.codes {
        let mut state = f64::NAN;
        for d in 0..shape.dates {
            let v = x[shape.at(d, c)];
            if v.is_finite() {
                state = if state.is_finite() {
                    alpha * v + (1.0 - alpha) * state
                } else {
                    v
                };
                out[shape.at(d, c)] = state
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gaps_are_not_compressed() {
        let s = Shape { dates: 4, codes: 1 };
        let x = vec![1.0, f64::NAN, 3.0, 4.0];
        let y = rolling(&s, &x, 2, Rolling::Mean);
        assert!(y[2].is_nan());
        assert_eq!(y[3], 3.5)
    }
    #[test]
    fn ranks_use_min_ties() {
        let s = Shape { dates: 1, codes: 3 };
        assert_eq!(
            cross_rank(&s, &[2.0, 1.0, 1.0]),
            vec![1.0, 1.0 / 3.0, 1.0 / 3.0]
        );
    }
    #[test]
    fn recursive_ewm_is_adjust_false() {
        let s = Shape { dates: 3, codes: 1 };
        assert_eq!(ewm(&s, &[1.0, 3.0, 5.0], 0.5), vec![1.0, 2.0, 3.5]);
    }
}
