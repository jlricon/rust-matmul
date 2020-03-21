use tch::{Device, Kind, Tensor};

fn matmul_torch(n: i64, m: i64) -> Tensor {
    dbg!(tch::Cuda::is_available());
    let a = Tensor::ones(&[n, m], (Kind::Float, Device::Cuda(0))) * 0.1;
    let b = Tensor::ones(&[m, n], (Kind::Float, Device::Cuda(0))) * 0.1;
    let o = a.matmul(&b);
    o
}
fn main() {
    println!("{:?}", matmul_torch(100, 200));
}
