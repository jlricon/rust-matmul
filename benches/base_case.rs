use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
type Matrix = Vec<Vec<f32>>;
use matrixmultiply::sgemm;
use nalgebra::{ArrayStorage, Dynamic, Matrix as NMatrix, VecStorage};
use nalgebra::{U100, U20};
use ndarray::Array;
use tch::{Device, Kind, Tensor};
type DMatrix = NMatrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;
type SMatrix1000 = NMatrix<f32, U1000, U1000, ArrayStorage<f32, U1000, U1000>>;
type SMatrix100 = NMatrix<f32, U100, U100, ArrayStorage<f32, U100, U100>>;
type SMatrix200 = NMatrix<f32, U200, U200, ArrayStorage<f32, U200, U200>>;
type SMatrix20 = NMatrix<f32, U20, U20, ArrayStorage<f32, U20, U20>>;
use typenum::{U1000, U200};
fn get_matrices(n: usize) -> (Matrix, Matrix) {
    let a: Matrix = vec![vec![0.1; n]; n];
    let b: Matrix = vec![vec![0.1; n]; n];
    (a, b)
}
fn matmul_torch(n: i64) {
    let a = Tensor::ones(&[n, n], (Kind::Float, Device::Cpu)) * 0.1;
    let b = Tensor::ones(&[n, n], (Kind::Float, Device::Cpu)) * 0.1;
    let o = a * b;
}
fn matmul_nalgebra(n: usize) {
    let a = DMatrix::from_element(n, n, 0.1);
    let b = DMatrix::from_element(n, n, 0.1);
    let o = a.dot(&b);
}
fn matmul_nalgebra_static(n: usize) {
    match n {
        1000 => {
            let a = SMatrix1000::repeat(0.1);
            let b = SMatrix1000::repeat(0.1);
            let o = a.dot(&b);
        }
        100 => {
            let a = SMatrix100::repeat(0.1);
            let b = SMatrix100::repeat(0.1);
            let o = a.dot(&b);
        }
        20 => {
            let a = SMatrix20::repeat(0.1);
            let b = SMatrix20::repeat(0.1);
            let o = a.dot(&b);
        }
        200 => {
            let a = SMatrix200::repeat(0.1);
            let b = SMatrix200::repeat(0.1);
            let o = a.dot(&b);
        }
        _ => unimplemented!(),
    }
}
fn matmul_baseline_static(n: usize) {
    match n {
        100 => {
            let a = [[0.1 as f32; 100]; 100];
            let b = [[0.1 as f32; 100]; 100];
            let o = matmul_static(&a, &b);
        }

        _ => unimplemented!(),
    }
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
fn matmul_static(a: &[[f32; 100]; 100], b: &[[f32; 100]; 100]) -> [[f32; 100]; 100] {
    let mut out = [[0.0 as f32; 100]; 100];
    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                unsafe {
                    *out.get_unchecked_mut(i).get_unchecked_mut(j) +=
                        a.get_unchecked(i).get_unchecked(k) * b.get_unchecked(k).get_unchecked(j);
                }
            }
        }
    }
    out
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
        group.bench_with_input(BenchmarkId::new("NalgebraStatic", n), n, |b, &n| {
            b.iter(|| matmul_nalgebra_static(n));
        });
        group.bench_with_input(BenchmarkId::new("Torch", n), n, |b, &n| {
            b.iter(|| matmul_torch(n as i64));
        });
        if *n == 100 {
            // Otherwise we get stack overflow
            group.bench_with_input(BenchmarkId::new("UncheckedStatic", n), n, |b, &n| {
                b.iter(|| matmul_baseline_static(n));
            });
        }
    }

    group.finish();
}

fn alternate_measurement() -> Criterion {
    Criterion::default().sample_size(20)
}
criterion_group!(name=benches;
                config = alternate_measurement();
                targets=base_matmul);
criterion_main!(benches);
