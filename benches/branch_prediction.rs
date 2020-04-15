use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
fn if_in_loop(n: usize) {
    let arr1: Vec<f32> = vec![0.1; n];
    let arr2 = vec![0.2; n];
    let res: Vec<f32> = arr1
        .iter()
        .zip(arr2.iter())
        .enumerate()
        .map(|(p, (a, b))| if p == 10 { a * 10.0 + b } else { a + b })
        .collect();
}
fn clean_loop(n: usize) {
    let arr1: Vec<f32> = vec![0.1; n];
    let arr2 = vec![0.2; n];
    let mut res: Vec<f32> = arr1.iter().zip(arr2.iter()).map(|(a, b)| a + b).collect();
    res[10] = arr1[10] * 10.0 + arr2[10];
}
fn branch_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("branch_prediction");
    for n in [100, 200, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("Clean loop", n), n, |b, &n| {
            b.iter(|| clean_loop(n));
        });

        group.bench_with_input(BenchmarkId::new("If in loop", n), n, |b, &n| {
            b.iter(|| if_in_loop(n));
        });
    }

    group.finish();
}
fn alternate_measurement() -> Criterion {
    Criterion::default()
}
criterion_group!(name=benches;
                config = alternate_measurement();
                targets=branch_prediction);
criterion_main!(benches);
