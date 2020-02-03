# declare -a path=(
#                   "cifar10/scale_exp/"
#                   "cifar100/scale_exp/"
#                 )
# declare -a args_version=("scale_exp 0")


declare -a path=( # "mnist"
                  "cifar10"
                  # "cifar100"
                  # "scale_exp"
                  # "alpha_beta_exp"
                )

declare -a args_version=(
                         "run1 0"
                         "run2 123"
                         "run3 515"
                         # "scale_exp 123"
                         # "alpha_beta_exp 123"
                         )


#
for i in "${args_version[@]}"
do
  for j in "${path[@]}"
  do
    cd $j
    target="./all.sh ${i}"
    echo $j $target
    $target
    cd ../
  done
done
