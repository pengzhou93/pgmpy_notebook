#!/bin/bash

debug_str="import pydevd;pydevd.settrace('localhost', port=8081, stdoutToServer=True, stderrToServer=True)"
# pydevd module path
export PYTHONPATH=/home/shhs/Desktop/user/soft/pycharm-2018.1.4/debug-eggs/pycharm-debug-py3k.egg_FILES

insert_debug_string()
{
file=$1
line=$2
debug_string=$3
debug=$4

value=`sed -n ${line}p "$file"`
if [ "$value" != "$debug_str" ] && [ "$debug" = debug ]
then
echo "++Insert $debug_string in line_${line}++"
sed -i "${line}i $debug_str" "$file"
fi
}

delete_debug_string()
{
file=$1
line=$2
debug_string=$3

value=`sed -n ${line}p "$file"`
if [ "$value" = "$debug_str" ]
then
echo "--Delete $debug_string in line_${line}--"
sed -i "${line}d" "$file"
fi
}

if [ "$1" = 'notebooks/1. Introduction to Probabilistic Graphical Models.ipynb' ]
then
    # ./run.sh "notebooks/1. Introduction to Probabilistic Graphical Models.ipynb" debug

    # python3.6 tf_1_6
    # edward: pip install edward
    source $HOME/anaconda3/bin/activate tf_1_6
    export LD_LIBRARY_PATH=/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

    debug=$2
    if [ $debug = debug ]
    then
        cd notebooks
        file="1. Introduction to Probabilistic Graphical Models.py"
        line=2
        insert_debug_string "$file" $line "$debug_str" $debug
        python "$file"
        delete_debug_string "$file" $line "$debug_str"
    else
        jupyter notebook --browser google-chrome
    fi

elif [ "$1" = 'notebooks/2. Bayesian Networks.ipynb' ]
then
    # ./run.sh "notebooks/2. Bayesian Networks.ipynb" debug

    # python3.6 tf_1_6
    # edward: pip install edward
    source $HOME/anaconda3/bin/activate tf_1_6
    export LD_LIBRARY_PATH=/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

    debug=$2
    if [ $debug = debug ]
    then
        cd notebooks
        file="2. Bayesian Networks.ipynb"
        line=2
        insert_debug_string "$file" $line "$debug_str" $debug
        python "$file"
        delete_debug_string "$file" $line "$debug_str"
    else
        jupyter notebook --browser google-chrome
    fi

else
    echo "No parameters"
fi