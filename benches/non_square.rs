use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
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

fn matmul_ndarray(n: usize) {
    let a = Array::<f32, _>::from_elem((n, n), 0.1);
    let b = Array::<f32, _>::from_elem((n, n), 0.1);
    let out = a.dot(&b);
}

fn base_matmul(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("non_square");
    group.plot_config(plot_config);
    for n in [20, 100, 200, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("Ndarray", n), n, |b, &n| {
            b.iter(|| matmul_ndarray(n));
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
    }

    group.finish();
}

fn alternate_measurement() -> Criterion {
    Criterion::default()
}
criterion_group!(name=benches;
                config = alternate_measurement();
                targets=base_matmul);
criterion_main!(benches);
