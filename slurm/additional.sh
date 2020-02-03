
# declare -a path=(
#                   "mnist"
#                 )
#
# declare -a args_version=(
#                          "run1 0"
#                          "run2 123"
#                          "run3 515"
#                          # "scale_exp 123"
#                          # "alpha_beta_exp 123"
#                          )

# declare -a run_version=(
#                           "run1"
#                           "run2"
#                           "run3"
#                         )
#
declare -A args_version=(
                         [run1]=0
                         [run2]=123
                         [run3]=515
                         )
#
cd ..

declare -a nr_arr=("0.0"
                   "0.2"
                   "0.4"
                   "0.6"
                   "0.8")

#
for run_version in "${!args_version[@]}"; do
  seed="${args_version[$run_version]}"
  echo "seed is $seed and run_version is $run_version"
  for i in "${nr_arr[@]}"
      do
        python3 -u train.py   --nr              $i                  \
                              --loss            NLNL                \
                              --run_version     $run_version        \
                              --dataset_type    mnist               \
                              --seed            $seed
  done
done

declare -a nr_arr=("0.1"
                   "0.2"
                   "0.3"
                   "0.4")

#
for run_version in "${!args_version[@]}"; do
  seed="${args_version[$run_version]}"
  echo "seed is $seed and run_version is $run_version"
  for i in "${nr_arr[@]}"
      do
        python3 -u train.py   --nr              $i                  \
                              --loss            NLNL                \
                              --asym                                \
                              --run_version     $run_version        \
                              --dataset_type    mnist               \
                              --seed            $seed
  done
done
