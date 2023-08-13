conda activate BT

for n_joints in 10 15 20 25
do
    echo $n_joints
    python -m latent.datasets.create_dataset $n_joints 10000 IK_random_start train
    python -m latent.datasets.create_dataset $n_joints 2000 IK_random_start val
    python -m latent.datasets.create_dataset $n_joints 1000 IK_random_start test
done
