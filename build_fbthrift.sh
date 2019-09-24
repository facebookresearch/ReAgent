set -e

if [[ "$OSTYPE" == "darwin"* ]]; then
  # Mac OSX
  brew install \
    cmake \
    boost \
    double-conversion \
    gflags \
    glog \
    libevent \
    lz4 \
    snappy \
    xz \
    openssl \
    libsodium

  brew link \
    boost \
    double-conversion \
    gflags \
    glog \
    libevent \
    lz4 \
    snappy \
    xz \
    libsodium

  export OPENSSL_LIBRARIES=/usr/local/opt/openssl/lib
  export OPENSSL_ROOT_DIR=/usr/local/opt/openssl
  export BISON_EXECUTABLE=/usr/local/opt/bison/bin/bison
else
  export OPENSSL_LIBRARIES=
  export OPENSSL_ROOT_DIR=
  export BISON_EXECUTABLE=/usr/bin/bison
fi

echo "We need sudo privledges to install libraries"
sudo ls > /dev/null

pushd external/fmt
cmake -G "Unix Makefiles" .
make -j`nproc`
sudo make install
popd

pushd external/folly
cmake -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR} -DOPENSSL_LIBRARIES=${OPENSSL_LIBRARIES} -G "Unix Makefiles" .
make -j`nproc`
sudo make install
popd

mkdir -p external/rsocket-cpp/cmake_build
pushd external/rsocket-cpp/cmake_build
cmake -G "Unix Makefiles" -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR} -DOPENSSL_LIBRARIES=${OPENSSL_LIBRARIES} -DBUILD_TESTS=OFF ..
make -j`nproc`
sudo make install
popd

mkdir -p external/fizz/fizz/build
pushd external/fizz/fizz/build
cmake -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR} -DOPENSSL_LIBRARIES=${OPENSSL_LIBRARIES} -G "Unix Makefiles" ..
make -j`nproc`
sudo make install
popd

pushd external/wangle/wangle
cmake -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR} -DOPENSSL_LIBRARIES=${OPENSSL_LIBRARIES} -G "Unix Makefiles" .
make -j`nproc`
sudo make install
popd

pushd external/fbthrift/build
cmake -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR} -DOPENSSL_LIBRARIES=${OPENSSL_LIBRARIES} -DBISON_EXECUTABLE=${BISON_EXECUTABLE} -G "Unix Makefiles" ..
make -j`nproc`
sudo make install
popd

