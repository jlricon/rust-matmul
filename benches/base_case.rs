use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
type Matrix = Vec<Vec<f32>>;
use matrixmultiply::sgemm;
use nalgebra::{Dynamic, Matrix as NMatrix, VecStorage};
use ndarray::{Array, ArrayBase};
type DMatrix = NMatrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;
fn get_matrices(n: usize) -> (Matrix, Matrix) {
    let a: Matrix = vec![vec![0.1; n]; n];
    let b: Matrix = vec![vec![0.1; n]; n];
    (a, b)
}
fn matmul_nalgebra(n: usize) {
    let a = DMatrix::from_element(n, n, 0.1);
    let b = DMatrix::from_element(n, n, 0.1);
    let o = a.dot(&b);
}
fn matmul_matrixmultiply(n: usize) {
    let a = vec![0.1; n * n];
    let b = vec![0.1; n * n];
    let mut c = vec![0.0; n * n];
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();
    unsafe {
        sgemm(
            n, n, n, 1.0, a_ptr, n as isize, 1, b_ptr, n as isize, 1, 0.0, c_ptr, n as isize, 1,
        )
    }
}
fn matmul_ndarray(n: usize) {
    let a = Array::<f32, _>::from_elem((n, n), 0.1);
    let b = Array::<f32, _>::from_elem((n, n), 0.1);
    let out = a.dot(&b);
}
fn matmul(n: usize) {
    let (a, b) = get_matrices(n);
    let mut out: Matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
fn matmul_opt1(n: usize) {
    let (a, b) = get_matrices(n);
    let mut out: Matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                unsafe {
                    *out.get_unchecked_mut(i).get_unchecked_mut(j) +=
                        a.get_unchecked(i).get_unchecked(k) * b.get_unchecked(k).get_unchecked(j);
                }
            }
        }
    }
}
fn base_matmul(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("base_case");
    group.plot_config(plot_config);
    for n in [20, 100, 200, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("Base", n), n, |b, &n| {
            b.iter(|| matmul(n));
        });
        group.bench_with_input(BenchmarkId::new("Unchecked", n), n, |b, &n| {
            b.iter(|| matmul_opt1(n));
        });
        group.bench_with_input(BenchmarkId::new("Ndarray", n), n, |b, &n| {
            b.iter(|| matmul_ndarray(n));
        });
        group.bench_with_input(BenchmarkId::new("Matrixmultiply", n), n, |b, &n| {
            b.iter(|| matmul_matrixmultiply(n));
        });
        group.bench_with_input(BenchmarkId::new("Nalgebra", n), n, |b, &n| {
            b.iter(|| matmul_nalgebra(n));
        });
    }
    group.finish();
}

fn alternate_measurement() -> Criterion {
    Criterion::default().sample_size(10)
}
criterion_group!(name=benches;
                config = alternate_measurement();
                targets=base_matmul);
criterion_main!(benches);
