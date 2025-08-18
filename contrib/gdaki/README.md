## Usage
### Initial

```
cp -r ./ucx-fork/contrib/gdaki ./

cd gdaki
source ./env.sh
./build.sh

sudo su
source ./env.sh
cd ucx
./test/gtest/gtest --gtest_filter=*gdaki*:*ucp_batch*

```
### Incremental
```
ssh rock0X
cd gdaki
source ./env.sh
sudo su
rock01$ ./run_sample.sh -c 1.1.60.10
rock10$ ./run_sample.sh

```
