use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};

fn array_10000(arr1: &[u8; 10000], arr2: &mut [bool; 10000]) {
    arr1.iter()
        .zip(arr2.iter_mut())
        .for_each(|(a, b)| *b = *a == 2);
}
fn array_100000(arr1: &[u8; 100000], arr2: &mut [bool; 100000]) {
    arr1.iter()
        .zip(arr2.iter_mut())
        .for_each(|(a, b)| *b = *a == 2);
}
fn array_512(arr1: &[u8; 512], arr2: &mut [bool; 512]) {
    arr1.iter()
        .zip(arr2.iter_mut())
        .for_each(|(a, b)| *b = *a == 2);
}
fn array_64(arr1: &[u8; 64], arr2: &mut [bool; 64]) {
    arr1.iter()
        .zip(arr2.iter_mut())
        .for_each(|(a, b)| *b = *a == 2);
}

fn vector_realloc(arr1: &Vec<u8>) {
    let res: Vec<bool> = arr1.iter().map(|a| *a == 2).collect();
}
fn vector_no_realloc(arr1: &Vec<u8>, arr2: &mut Vec<bool>) {
    arr1.iter().zip(arr2).for_each(|(a, b)| *b = *a == 2);
}

fn simd_kinds(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    // Arrays
    let base_array = [10; 512];
    let mut return_array = [false; 512];
    let base_array64 = [10; 64];
    let mut return_array64 = [false; 64];
    let base_array10k = [10; 10000];
    let mut return_array10k = [false; 10000];
    let base_array100k = [10; 100000];
    let mut return_array100k = [false; 100000];
    // End of arrays
    let mut group = c.benchmark_group("simd_kinds");
    group.plot_config(plot_config);
    for n in [32, 64, 100, 128, 256, 10000, 100000].iter() {
        // Vectors of varying lengths
        let base_vector = vec![10; *n];
        let mut return_vector = vec![false; *n];
        group.bench_with_input(BenchmarkId::new("Vector with realloc", n), n, |b, &n| {
            b.iter(|| vector_realloc(&base_vector));
        });

        group.bench_with_input(BenchmarkId::new("Vector with no realloc", n), n, |b, &n| {
            b.iter(|| vector_no_realloc(&base_vector, &mut return_vector));
        });

        if *n == 512 {
            group.bench_with_input(BenchmarkId::new("Array with no realloc", n), n, |b, &n| {
                b.iter(|| array_512(&base_array, &mut return_array));
            });
        }
        if *n == 64 {
            group.bench_with_input(BenchmarkId::new("Array with no realloc", n), n, |b, &n| {
                b.iter(|| array_64(&base_array64, &mut return_array64));
            });
        }
        if *n == 10000 {
            group.bench_with_input(BenchmarkId::new("Array with no realloc", n), n, |b, &n| {
                b.iter(|| array_10000(&base_array10k, &mut return_array10k));
            });
        }
        if *n == 100000 {
            group.bench_with_input(BenchmarkId::new("Array with no realloc", n), n, |b, &n| {
                b.iter(|| array_100000(&base_array100k, &mut return_array100k));
            });
        }
    }

    group.finish();
}
fn alternate_measurement() -> Criterion {
    Criterion::default()
}
criterion_group!(name=benches;
                config = alternate_measurement();
                targets=simd_kinds);
criterion_main!(benches);
