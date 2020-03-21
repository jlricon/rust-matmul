use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use nalgebra::{ArrayStorage, Dynamic, Matrix as NMatrix, VecStorage};
use nalgebra::{U100, U20};
use ndarray::Array;
use tch::{Device, Kind, Tensor};
type DMatrix = NMatrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;
use typenum::{U1000, U200};

fn matmul_torch(n: i64, m: i64) {
    let a = Tensor::ones(&[n, m], (Kind::Float, Device::Cuda(0))) * 0.1;
    let b = Tensor::ones(&[m, n], (Kind::Float, Device::Cuda(0))) * 0.1;
    let o = a.matmul(&b);
}
fn matmul_nalgebra(n: usize, m: usize) {
    let a = DMatrix::from_element(n, m, 0.1);
    let b = DMatrix::from_element(m, n, 0.1);
    let o = a * b;
}

fn matmul_ndarray(n: usize, m: usize) {
    let a = Array::<f32, _>::from_elem((n, m), 0.1);
    let b = Array::<f32, _>::from_elem((m, n), 0.1);
    let out = a.dot(&b);
}

fn base_matmul(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("non_square");
    group.plot_config(plot_config);
    for n in [100, 200, 500, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("Ndarray", n), n, |b, &n| {
            b.iter(|| matmul_ndarray(n, 200));
        });

        group.bench_with_input(BenchmarkId::new("Nalgebra", n), n, |b, &n| {
            b.iter(|| matmul_nalgebra(n, 200));
        });

        group.bench_with_input(BenchmarkId::new("Torch", n), n, |b, &n| {
            b.iter(|| matmul_torch(n as i64, 200));
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
