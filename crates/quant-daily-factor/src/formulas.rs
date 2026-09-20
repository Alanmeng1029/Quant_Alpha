use crate::engine::*;
use crate::panel::Panel;

fn constant(n: usize, v: f64) -> Series {
    vec![v; n]
}
fn add(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| x + y)
}
fn sub(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| x - y)
}
fn mul(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| x * y)
}
fn div(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| if y != 0.0 { x / y } else { f64::NAN })
}
fn neg(a: &[f64]) -> Series {
    unary(a, |x| -x)
}
fn abs(a: &[f64]) -> Series {
    unary(a, f64::abs)
}
fn gt(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| (x > y) as u8 as f64)
}
fn ge(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| (x >= y) as u8 as f64)
}
fn lt(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| (x < y) as u8 as f64)
}
fn le(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, |x, y| (x <= y) as u8 as f64)
}
fn max2(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, f64::max)
}
fn min2(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, f64::min)
}
fn pow(a: &[f64], b: &[f64]) -> Series {
    binary(a, b, f64::powf)
}
fn fill(a: &[f64], value: f64) -> Series {
    a.iter()
        .map(|x| if x.is_finite() { *x } else { value })
        .collect()
}
fn weighted2(s: &Shape, a: &[f64]) -> Series {
    scalar(
        &add(&delay(s, a, 1), &scalar(a, 2.0, |x, y| x * y)),
        3.0,
        |x, y| x / y,
    )
}

pub fn implemented_names() -> Vec<&'static str> {
    crate::registry::FACTOR_NAMES.to_vec()
}

pub fn compute(name: &str, p: &Panel) -> Option<Series> {
    let s = &p.shape;
    let n = s.len();
    Some(match name {
        "gtja_alpha159_raw_v1" => {
            let pc = delay(s, &p.close, 1);
            let lo = min2(&p.low, &pc);
            let hi = max2(&p.high, &pc);
            let range = sub(&hi, &lo);
            let part = |w: usize, m: f64| {
                scalar(
                    &div(
                        &sub(&p.close, &rolling(s, &lo, w, Rolling::Sum)),
                        &rolling(s, &range, w, Rolling::Sum),
                    ),
                    m,
                    |x, y| x * y,
                )
            };
            scalar(
                &add(&add(&part(6, 288.0), &part(12, 144.0)), &part(24, 144.0)),
                100.0 / 504.0,
                |x, y| x * y,
            )
        }
        "gtja_alpha158_raw_v1" => div(&sub(&p.high, &p.low), &p.close),
        "gtja_alpha042_raw_v1" | "wq_alpha040_raw_v1" => neg(&mul(
            &cross_rank(s, &rolling(s, &p.high, 10, Rolling::Std)),
            &corr(s, &p.high, &p.volume, 10),
        )),
        "gtja_alpha187_raw_v1" => {
            let po = delay(s, &p.open, 1);
            let zero = constant(n, 0.0);
            let signal = choose(
                &gt(&p.open, &po),
                &max2(&sub(&p.high, &p.open), &sub(&p.open, &po)),
                &zero,
            );
            rolling(s, &signal, 20, Rolling::Sum)
        }
        "gtja_alpha175_raw_v1" | "gtja_alpha161_raw_v1" => {
            let pc = delay(s, &p.close, 1);
            let tr = max2(
                &sub(&p.high, &p.low),
                &max2(&abs(&sub(&pc, &p.high)), &abs(&sub(&pc, &p.low))),
            );
            rolling(
                s,
                &tr,
                if name.contains("175") { 6 } else { 12 },
                Rolling::Mean,
            )
        }
        "gtja_alpha189_raw_v1" => {
            let m = rolling(s, &p.close, 6, Rolling::Mean);
            rolling(s, &abs(&sub(&p.close, &m)), 6, Rolling::Mean)
        }
        "gtja_alpha167_raw_v1" => {
            let d = delta(s, &p.close, 1);
            rolling(
                s,
                &choose(&gt(&d, &constant(n, 0.0)), &d, &constant(n, 0.0)),
                12,
                Rolling::Sum,
            )
        }
        "gtja_alpha174_raw_v1" => {
            let pc = delay(s, &p.close, 1);
            let st = rolling(s, &p.close, 20, Rolling::Std);
            ewm(
                s,
                &choose(&gt(&p.close, &pc), &st, &constant(n, 0.0)),
                1.0 / 20.0,
            )
        }
        "gtja_alpha070_raw_v1" => rolling(s, &mul(&p.volume, &p.vwap), 6, Rolling::Std),
        "gtja_alpha185_raw_v1" => {
            let r = div(&p.open, &p.close);
            cross_rank(s, &neg(&unary(&sub(&constant(n, 1.0), &r), |x| x * x)))
        }
        "wq_alpha083_raw_v1" => {
            let range = div(
                &sub(&p.high, &p.low),
                &rolling(s, &p.close, 5, Rolling::Mean),
            );
            let a = cross_rank(s, &delay(s, &range, 2));
            let b = cross_rank(s, &cross_rank(s, &p.volume));
            mul(&a, &div(&b, &div(&range, &sub(&p.vwap, &p.close))))
        }
        "gtja_alpha129_raw_v1" => {
            let d = delta(s, &p.close, 1);
            rolling(
                s,
                &choose(&lt(&d, &constant(n, 0.0)), &abs(&d), &constant(n, 0.0)),
                12,
                Rolling::Sum,
            )
        }
        "gtja_alpha160_raw_v1" => {
            let pc = delay(s, &p.close, 1);
            let st = rolling(s, &p.close, 20, Rolling::Std);
            ewm(
                s,
                &choose(&le(&p.close, &pc), &st, &constant(n, 0.0)),
                1.0 / 20.0,
            )
        }
        "gtja_alpha093_raw_v1" => {
            let po = delay(s, &p.open, 1);
            let zero = constant(n, 0.0);
            let signal = choose(
                &lt(&p.open, &po),
                &max2(&sub(&p.open, &p.low), &sub(&p.open, &po)),
                &zero,
            );
            rolling(s, &signal, 20, Rolling::Sum)
        }
        "gtja_alpha126_raw_v1" => scalar(&add(&add(&p.close, &p.high), &p.low), 3.0, |x, y| x / y),
        "gtja_alpha153_raw_v1" => scalar(
            &add(
                &add(
                    &rolling(s, &p.close, 3, Rolling::Mean),
                    &rolling(s, &p.close, 6, Rolling::Mean),
                ),
                &add(
                    &rolling(s, &p.close, 12, Rolling::Mean),
                    &rolling(s, &p.close, 24, Rolling::Mean),
                ),
            ),
            4.0,
            |x, y| x / y,
        ),
        "wq_alpha041_raw_v1" => sub(&unary(&mul(&p.high, &p.low), f64::sqrt), &p.vwap),
        "wq_alpha005_raw_v1" => {
            let a = neg(&cross_rank(
                s,
                &sub(&p.open, &rolling(s, &p.vwap, 10, Rolling::Mean)),
            ));
            let b = abs(&cross_rank(s, &sub(&p.close, &p.vwap)));
            mul(&a, &b)
        }
        "gtja_alpha095_raw_v1" => rolling(s, &mul(&p.volume, &p.vwap), 20, Rolling::Std),
        "gtja_alpha120_raw_v1" | "wq_alpha042_raw_v1" => div(
            &cross_rank(s, &sub(&p.vwap, &p.close)),
            &cross_rank(s, &add(&p.vwap, &p.close)),
        ),
        "gtja_alpha150_raw_v1" => mul(
            &scalar(&add(&add(&p.close, &p.high), &p.low), 3.0, |x, y| x / y),
            &p.volume,
        ),
        "gtja_alpha132_raw_v1" => rolling(s, &mul(&p.volume, &p.vwap), 20, Rolling::Mean),
        "wq_alpha006_raw_v1" => fill(&neg(&corr(s, &p.open, &p.volume, 10)), 0.0),
        "gtja_alpha139_raw_v1" => neg(&corr(s, &p.open, &p.volume, 10)),
        "gtja_alpha040_raw_v1" => {
            let pc = delay(s, &p.close, 1);
            let up = choose(&gt(&p.close, &pc), &p.volume, &constant(n, 0.0));
            let down = choose(&le(&p.close, &pc), &p.volume, &constant(n, 0.0));
            scalar(
                &div(
                    &rolling(s, &up, 26, Rolling::Sum),
                    &rolling(s, &down, 26, Rolling::Sum),
                ),
                100.0,
                |x, y| x * y,
            )
        }
        "wq_alpha088_raw_v1" => {
            let adv = rolling(s, &p.volume, 60, Rolling::Mean);
            let left = sub(
                &add(&cross_rank(s, &p.open), &cross_rank(s, &p.low)),
                &add(&cross_rank(s, &p.high), &cross_rank(s, &p.close)),
            );
            let p1 = cross_rank(s, &decay_linear(s, &left, 8));
            let p2 = ts_rank(
                s,
                &decay_linear(
                    s,
                    &corr(s, &ts_rank(s, &p.close, 8), &ts_rank(s, &adv, 21), 8),
                    7,
                ),
                3,
            );
            min2(&p1, &p2)
        }
        "gtja_alpha140_raw_v1" => {
            let adv = rolling(s, &p.volume, 60, Rolling::Mean);
            let left = sub(
                &add(&cross_rank(s, &p.open), &cross_rank(s, &p.low)),
                &add(&cross_rank(s, &p.high), &cross_rank(s, &p.close)),
            );
            let p1 = cross_rank(s, &decay_linear(s, &left, 8));
            let close_rank = scalar(&ts_rank(s, &p.close, 8), 8.0, |x, y| x / y);
            let adv_rank = scalar(&ts_rank(s, &adv, 20), 20.0, |x, y| x / y);
            let p2 = scalar(
                &ts_rank(
                    s,
                    &decay_linear(s, &corr(s, &close_rank, &adv_rank, 8), 7),
                    3,
                ),
                3.0,
                |x, y| x / y,
            );
            min2(&p1, &p2)
        }
        "gtja_alpha165_raw_v1" | "gtja_alpha183_raw_v1" => {
            let w = if name.contains("165") { 48 } else { 24 };
            let dev = sub(&p.close, &rolling(s, &p.close, w, Rolling::Mean));
            let sum = rolling(s, &dev, w, Rolling::Sum);
            sub(
                &cross_extreme(s, &sum, true),
                &div(
                    &cross_extreme(s, &sum, false),
                    &rolling(s, &p.close, w, Rolling::Std),
                ),
            )
        }
        "gtja_alpha114_raw_v1" => {
            let range = div(
                &sub(&p.high, &p.low),
                &rolling(s, &p.close, 5, Rolling::Mean),
            );
            mul(
                &cross_rank(s, &delay(s, &range, 2)),
                &div(
                    &cross_rank(s, &cross_rank(s, &p.volume)),
                    &div(&range, &sub(&p.vwap, &p.close)),
                ),
            )
        }
        "gtja_alpha164_raw_v1" => {
            let pc = delay(s, &p.close, 1);
            let d = sub(&p.close, &pc);
            let a = choose(
                &gt(&p.close, &pc),
                &div(&constant(n, 1.0), &d),
                &constant(n, 1.0),
            );
            let b = scalar(
                &div(
                    &sub(&a, &rolling(s, &a, 12, Rolling::Min)),
                    &sub(&p.high, &p.low),
                ),
                100.0,
                |x, y| x * y,
            );
            ewm(s, &b, 2.0 / 13.0)
        }
        "gtja_alpha121_raw_v1" => {
            let adv = rolling(s, &p.volume, 60, Rolling::Mean);
            let base = cross_rank(s, &sub(&p.vwap, &min2(&p.vwap, &constant(n, 12.0))));
            let exponent = scalar(
                &ts_rank(
                    s,
                    &corr(s, &ts_rank(s, &p.vwap, 20), &ts_rank(s, &adv, 2), 18),
                    3,
                ),
                3.0,
                |x, y| x / y,
            );
            neg(&pow(&base, &exponent))
        }
        "gtja_alpha010_raw_v1" => {
            let ret = sub(&div(&p.close, &delay(s, &p.close, 1)), &constant(n, 1.0));
            let st = rolling(s, &ret, 20, Rolling::Std);
            let base = choose(&lt(&ret, &constant(n, 0.0)), &st, &p.close);
            cross_rank(s, &rolling(s, &unary(&base, |x| x * x), 5, Rolling::Max))
        }
        "gtja_alpha173_raw_v1" => {
            let a = ewm(s, &p.close, 2.0 / 13.0);
            let b = ewm(s, &a, 2.0 / 13.0);
            add(
                &sub(
                    &scalar(&a, 3.0, |x, y| x * y),
                    &scalar(&b, 2.0, |x, y| x * y),
                ),
                &ewm(s, &b, 2.0 / 13.0),
            )
        }
        "gtja_alpha124_raw_v1" => {
            let r = cross_rank(s, &rolling(s, &p.close, 30, Rolling::Max));
            div(&sub(&p.close, &p.vwap), &weighted2(s, &r))
        }
        "gtja_alpha041_raw_v1" => neg(&cross_rank(
            s,
            &rolling(s, &delta(s, &p.vwap, 3), 5, Rolling::Max),
        )),
        "wq_alpha077_raw_v1" => {
            let mid = scalar(&add(&p.high, &p.low), 2.0, |x, y| x / y);
            let p1 = cross_rank(s, &decay_linear(s, &sub(&mid, &p.vwap), 20));
            let p2 = cross_rank(
                s,
                &decay_linear(
                    s,
                    &corr(s, &mid, &rolling(s, &p.volume, 40, Rolling::Mean), 3),
                    6,
                ),
            );
            min2(&p1, &p2)
        }
        "gtja_alpha091_raw_v1" => neg(&mul(
            &cross_rank(s, &sub(&p.close, &max2(&p.close, &constant(n, 5.0)))),
            &cross_rank(
                s,
                &corr(s, &rolling(s, &p.volume, 40, Rolling::Mean), &p.low, 5),
            ),
        )),
        "gtja_alpha108_raw_v1" => neg(&pow(
            &cross_rank(s, &sub(&p.high, &min2(&p.high, &constant(n, 2.0)))),
            &cross_rank(
                s,
                &corr(s, &p.vwap, &rolling(s, &p.volume, 120, Rolling::Mean), 6),
            ),
        )),
        "gtja_alpha127_raw_v1" => unary(
            &rolling(
                s,
                &unary(
                    &scalar(
                        &div(
                            &sub(&p.close, &max2(&p.close, &constant(n, 12.0))),
                            &max2(&p.close, &constant(n, 12.0)),
                        ),
                        100.0,
                        |x, y| x * y,
                    ),
                    |x| x * x,
                ),
                12,
                Rolling::Mean,
            ),
            f64::sqrt,
        ),
        "gtja_alpha098_raw_v1" => {
            let mean = rolling(s, &p.close, 100, Rolling::Mean);
            let change = div(&sub(&mean, &delay(s, &mean, 100)), &delay(s, &p.close, 100));
            choose(
                &le(&change, &constant(n, 0.05)),
                &neg(&sub(&p.close, &rolling(s, &p.close, 100, Rolling::Min))),
                &neg(&sub(&p.close, &delay(s, &p.close, 3))),
            )
        }
        "wq_alpha094_raw_v1" => {
            let adv = rolling(s, &p.volume, 60, Rolling::Mean);
            let base = cross_rank(s, &sub(&p.vwap, &rolling(s, &p.vwap, 12, Rolling::Min)));
            neg(&pow(
                &base,
                &ts_rank(
                    s,
                    &corr(s, &ts_rank(s, &p.vwap, 20), &ts_rank(s, &adv, 4), 18),
                    3,
                ),
            ))
        }
        "wq_alpha026_raw_v1" => neg(&rolling(
            s,
            &fill(
                &corr(s, &ts_rank(s, &p.volume, 5), &ts_rank(s, &p.high, 5), 5),
                0.0,
            ),
            3,
            Rolling::Max,
        )),
        "wq_alpha057_raw_v1" => neg(&div(
            &sub(&p.close, &p.vwap),
            &decay_linear(s, &cross_rank(s, &ts_argmax(s, &p.close, 30)), 2),
        )),
        "gtja_alpha005_raw_v1" => neg(&rolling(
            s,
            &corr(s, &ts_rank(s, &p.volume, 5), &ts_rank(s, &p.high, 5), 5),
            3,
            Rolling::Max,
        )),
        "wq_alpha071_raw_v1" => {
            let adv = rolling(s, &p.volume, 180, Rolling::Mean);
            let p1 = ts_rank(
                s,
                &decay_linear(
                    s,
                    &corr(s, &ts_rank(s, &p.close, 3), &ts_rank(s, &adv, 12), 18),
                    4,
                ),
                16,
            );
            let base = cross_rank(
                s,
                &sub(&add(&p.low, &p.open), &scalar(&p.vwap, 2.0, |x, y| x * y)),
            );
            let p2 = ts_rank(s, &decay_linear(s, &unary(&base, |x| x * x), 16), 4);
            max2(&p1, &p2)
        }
        "gtja_alpha119_raw_v1" => {
            let left = cross_rank(
                s,
                &decay_linear(
                    s,
                    &corr(
                        s,
                        &p.vwap,
                        &rolling(
                            s,
                            &rolling(s, &p.volume, 5, Rolling::Mean),
                            26,
                            Rolling::Sum,
                        ),
                        5,
                    ),
                    7,
                ),
            );
            let c = corr(
                s,
                &cross_rank(s, &p.open),
                &cross_rank(s, &rolling(s, &p.volume, 15, Rolling::Mean)),
                21,
            );
            let right = cross_rank(
                s,
                &decay_linear(s, &ts_rank(s, &min2(&c, &constant(n, 9.0)), 7), 8),
            );
            sub(&left, &right)
        }
        "wq_alpha014_raw_v1" => {
            let ret = sub(&div(&p.close, &delay(s, &p.close, 1)), &constant(n, 1.0));
            mul(
                &neg(&cross_rank(s, &delta(s, &ret, 3))),
                &fill(&corr(s, &p.open, &p.volume, 10), 0.0),
            )
        }
        "gtja_alpha136_raw_v1" => {
            let ret = sub(&div(&p.close, &delay(s, &p.close, 1)), &constant(n, 1.0));
            neg(&mul(
                &cross_rank(s, &sub(&ret, &delay(s, &ret, 3))),
                &corr(s, &p.open, &p.volume, 10),
            ))
        }
        "gtja_alpha148_raw_v1" => {
            let left = cross_rank(
                s,
                &corr(
                    s,
                    &p.open,
                    &rolling(
                        s,
                        &rolling(s, &p.volume, 60, Rolling::Mean),
                        9,
                        Rolling::Sum,
                    ),
                    6,
                ),
            );
            let right = cross_rank(s, &sub(&p.open, &rolling(s, &p.open, 14, Rolling::Min)));
            neg(&lt(&left, &right))
        }
        "gtja_alpha076_raw_v1" => {
            let signal = div(
                &abs(&sub(
                    &div(&p.close, &delay(s, &p.close, 1)),
                    &constant(n, 1.0),
                )),
                &p.volume,
            );
            div(
                &rolling(s, &signal, 20, Rolling::Std),
                &rolling(s, &signal, 20, Rolling::Mean),
            )
        }
        "gtja_alpha154_raw_v1" => lt(
            &sub(&p.vwap, &min2(&p.vwap, &constant(n, 16.0))),
            &corr(s, &p.vwap, &rolling(s, &p.volume, 180, Rolling::Mean), 18),
        ),
        "wq_alpha065_raw_v1" => {
            let adv = rolling(s, &p.volume, 60, Rolling::Mean);
            let price = add(
                &scalar(&p.open, 0.00817205, |x, y| x * y),
                &scalar(&p.vwap, 1.0 - 0.00817205, |x, y| x * y),
            );
            neg(&lt(
                &cross_rank(s, &corr(s, &price, &rolling(s, &adv, 9, Rolling::Mean), 6)),
                &cross_rank(s, &sub(&p.open, &rolling(s, &p.open, 14, Rolling::Min))),
            ))
        }
        "gtja_alpha092_raw_v1" => {
            let composite = add(
                &scalar(&p.close, 0.35, |x, y| x * y),
                &scalar(&p.vwap, 0.65, |x, y| x * y),
            );
            let p1 = cross_rank(s, &decay_linear(s, &delta(s, &composite, 2), 3));
            let p2 = scalar(
                &ts_rank(
                    s,
                    &decay_linear(
                        s,
                        &abs(&corr(
                            s,
                            &rolling(s, &p.volume, 180, Rolling::Mean),
                            &p.close,
                            13,
                        )),
                        5,
                    ),
                    15,
                ),
                15.0,
                |x, y| x / y,
            );
            neg(&max2(&p1, &p2))
        }
        "gtja_alpha149_raw_v1" => {
            let ir = sub(
                &div(&p.index_close, &delay(s, &p.index_close, 1)),
                &constant(n, 1.0),
            );
            let sr = sub(&div(&p.close, &delay(s, &p.close, 1)), &constant(n, 1.0));
            let down = lt(&p.index_close, &delay(s, &p.index_close, 1));
            let nulls = constant(n, f64::NAN);
            rolling_beta_skip_nulls(
                s,
                &choose(&down, &sr, &nulls),
                &choose(&down, &ir, &nulls),
                252,
                2,
            )
        }
        "wq_alpha098_raw_v1" => {
            let adv5 = rolling(s, &p.volume, 5, Rolling::Mean);
            let left = cross_rank(
                s,
                &decay_linear(
                    s,
                    &corr(s, &p.vwap, &rolling(s, &adv5, 26, Rolling::Mean), 5),
                    7,
                ),
            );
            let adv15 = rolling(s, &p.volume, 15, Rolling::Mean);
            let c = corr(s, &cross_rank(s, &p.open), &cross_rank(s, &adv15), 21);
            let right = cross_rank(s, &decay_linear(s, &ts_rank(s, &ts_argmin(s, &c, 9), 7), 8));
            sub(&left, &right)
        }
        "wq_alpha051_raw_v1" => {
            let inner = sub(
                &scalar(
                    &sub(&delay(s, &p.close, 20), &delay(s, &p.close, 10)),
                    10.0,
                    |x, y| x / y,
                ),
                &scalar(&sub(&delay(s, &p.close, 10), &p.close), 10.0, |x, y| x / y),
            );
            choose(
                &lt(&inner, &constant(n, -0.05)),
                &constant(n, 1.0),
                &neg(&delta(s, &p.close, 1)),
            )
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registry_names_are_unique() {
        let mut n = implemented_names();
        n.sort();
        n.dedup();
        assert_eq!(n.len(), 60)
    }
}
