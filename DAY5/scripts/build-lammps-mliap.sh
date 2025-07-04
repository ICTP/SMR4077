source ~/.bashrc

module load openmpi/4.1.6--nvhpc--24.3 fftw/3.3.10--openmpi--4.1.6--nvhpc--24.3  nvhpc/24.3   gcc/12.2.0

micromamba activate janus 
python3 -m pip install Cython

base=$PWD

if [[ ! -d $lammps ]]; then
  pushd $base
  git clone -b develop https://github.com/lammps/lammps
  popd
fi


which gcc 
which mpicxx

src="$base/lammps"
flav="mace-develop"
build="build-kokkos-$flav"
install=$base/lammps-mace
rm -rf $build

MPI_CXX=$(which mpicxx) cmake  -D CMAKE_INSTALL_PREFIX=$install \
        -D CMAKE_CXX_COMPILER=$src/lib/kokkos/bin/nvcc_wrapper   \
        -B$build   -S $src/cmake \
        -D CMAKE_BUILD_TYPE=Release     \
        -D BUILD_MPI=yes     -D BUILD_OMP=yes    \
        -D BUILD_SHARED_LIBS=yes     \
        -D LAMMPS_EXCEPTIONS=yes     \
        -D PKG_KOKKOS=yes  \
        -D Kokkos_ENABLE_CUDA=yes   \
        -D Kokkos_ARCH_NATIVE=yes  \
        -D Kokkos_ARCH_AMPERE80=yes   \
        -D Kokkos_ENABLE_SERIAL=yes \
        -D Kokkos_ENABLE_OPENMP=yes    \
        -D Kokkos_ENABLE_DEBUG=no     \
        -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=no   \
        -D CUDA_MPS_SUPPORT=yes  \
        -D Kokkos_ENABLE_CUDA_UVM=no \
        -D PKG_ML-IAP=yes \
        -D MLIAP_ENABLE_PYTHON=yes \
        -D PKG_ML-SNAP=yes   \
        -D PKG_PYTHON=yes  -D CMAKE_CXX_STANDARD=17

cmake --build $build -- -j8
cmake --install $build 

cd $build
make install-python

