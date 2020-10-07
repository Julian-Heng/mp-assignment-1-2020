#!/usr/bin/env bash

main()
{
    py="python3"
    get_source_script="../src/get_sources.py"
    appendix_1=("resize" "rotate" "harris" "histogram" "sift_keypoints")
    appendix_2=("hog" "sift_descriptors")
    appendix_3=("threshold" "rgb_mute" "connected_components")
    appendix_4=("kmeans")
    appendix_5=("__init__" "_handle_fname" "_grayscale" "_plt_savefig" "_sift" "_append_filename" "_write")
    appendix_6=("Task" "Task1" "Task1Dugong" "Task2" "Task2Diamond" "Task2Dugong" "Task3" "DiamondTask3" "DugongTask3" "Task4" "runner" "main")

    for i in {1..6}; do
        eval "${py}" "${get_source_script}" "\${appendix_${i}[@]}" > "appendix_${i}.txt"
    done
}

main
