conda activate BT

for num_joints in 2 5 10 15 20 25
do
    echo $num_joints
    python -m vae.data.create_dataset $num_joints 10000 IK_random_start train
    python -m vae.data.create_dataset $num_joints 2000 IK_random_start val
    python -m vae.data.create_dataset $num_joints 1000 IK_random_start test
done