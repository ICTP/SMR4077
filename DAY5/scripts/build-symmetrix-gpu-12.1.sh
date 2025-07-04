module load openmpi/4.1.6--nvhpc--23.11  cuda/12.1 cmake gcc/12.2.0 


module list
nvcc --version
gcc --version

base=$PWD
lammps=$base/lammps
build=$base/build-symmetrix-gpu-121
install=$base/lammps-symmetrix-gpu-121
symmetrix=$base/symmetrix

if [[ ! -d $lammps ]]; then
  pushd $base
  git clone -b develop https://github.com/lammps/lammps
  popd
fi

if [[ ! -d $symmetrix ]]; then
  pushd $base
  git clone --recursive https://github.com/wcwitt/symmetrix
  popd
fi

if [ ! -L $lammps/src/KOKKOS/pair_symmetrix_mace_kokkos.cpp ]; then
  pushd $symmetrix/pair_symmetrix
  ./install.sh $lammps
  popd
fi


rm -rf $build
rm -rf $install


# build lammps
cmake \
    -B $build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DJSON_HAS_RANGES=0 -DFMT_USE_NONTYPE_TEMPLATE_ARGS=0 -march=native -ffast-math" \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D BUILD_OMP=ON \
    -D BUILD_MPI=ON \
    -D SYMMETRIX_KOKKOS=ON \
    -D CMAKE_CXX_COMPILER=$lammps/lib/kokkos/bin/nvcc_wrapper \
    -D BUILD_SHARED_LIBS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_AMPERE80=ON \
    -D SYMMETRIX_SPHERICART_CUDA=ON \
    -D Kokkos_ENABLE_DEBUG=no    \
    -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=no   \
    -D Kokkos_ENABLE_CUDA_UVM=yes \
    -D PKG_EXTRA-FIX=yes \
    -D PKG_EXTRA-PAIR=yes \
    -D Kokkos_ARCH_NATIVE=ON \
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -S $lammps/cmake \
    -D CMAKE_INSTALL_PREFIX=$install

cmake --build $build -- VERBOSE=1 -j$1
cmake --install $build

