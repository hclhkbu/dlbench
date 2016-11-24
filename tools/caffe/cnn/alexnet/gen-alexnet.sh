epoch_size=50000
network_name=alexnet
gpu_count="${gpu_count:-1}"
learning_rate="${learning_rate:-0.05}"
batch_size="${batch_size:-1024}"
num_epochs="${num_epochs:-40}"
batches_per_epoch=`awk "BEGIN {print int( (${epoch_size}+${batch_size}-1)/${batch_size} )}"` #50000/32 
max_iter=`awk "BEGIN {print int( ${batches_per_epoch}*${num_epochs} )}"` #50000/32 * 40
test_interval=`awk "BEGIN {print int((${epoch_size}+${batch_size}-1)/${batch_size}*2)}"` # 50000/32 * 2
device=GPU
model_file=${network_name}-b${batch_size}.prototxt
solver_file=${network_name}-b${batch_size}-${device}-solver.prototxt
cp ${network_name}.prototxt ${model_file}
sed -i -e "s/BATCHSIZE/${batch_size}/g" ${model_file}
sed -i -e "s|HOME|${HOME}|g" ${model_file}

cp solver.prototxt ${solver_file}
sed -i -e "s/BATCHSIZE/${batch_size}/g" ${solver_file}
sed -i -e "s/MAXITER/${max_iter}/g" ${solver_file}
sed -i -e "s/DEVICE/${device}/g" ${solver_file}
sed -i -e "s/TESTINTERVAL/${test_interval}/g" ${solver_file}

solver_file=${network_name}-b${batch_size}-${device}-solver${gpu_count}.prototxt
cp solver.prototxt ${solver_file}
mbatches_per_epoch=`awk "BEGIN {print int( (${epoch_size}+${batch_size}*${gpu_count}-1)/(${batch_size}*${gpu_count}) )}"` #50000/32 
mmax_iter=`awk "BEGIN {print int( ${mbatches_per_epoch}*${num_epochs} )}"` 
mdisplay_interval=1 #`awk "BEGIN {print int( ${mbatches_per_epoch}/4 )}"` # 50000 / 32 
mtest_interval=`awk "BEGIN {print int( ${mbatches_per_epoch} )}"` # 50000/32
sed -i -e "s/BATCHSIZE/${batch_size}/g" ${solver_file}
sed -i -e "s/MAXITER/${mmax_iter}/g" ${solver_file}
sed -i -e "s/DEVICE/${device}/g" ${solver_file}
sed -i -e "s/TESTINTERVAL/${mtest_interval}/g" ${solver_file}
sed -i -e "s/DISPLAYINTERVAL/${mdisplay_interval}/g" ${solver_file}

device=CPU
solver_file=${network_name}-b${batch_size}-${device}-solver.prototxt
cp solver.prototxt ${solver_file}
sed -i -e "s/BATCHSIZE/${batch_size}/g" ${solver_file}
sed -i -e "s/MAXITER/${max_iter}/g" ${solver_file}
sed -i -e "s/DEVICE/${device}/g" ${solver_file}
sed -i -e "s/TESTINTERVAL/${test_interval}/g" ${solver_file}
