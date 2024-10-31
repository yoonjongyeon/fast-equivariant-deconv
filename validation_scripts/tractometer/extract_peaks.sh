#!/bin/bash

# Activate environment
conda activate hsd

# Yaml parser
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}


# Extract peaks
eval $(parse_yaml $1)
echo ${data_data_path} ${testing_expname}
sh2peaks -num 10 -mask ${data_data_path}/mask.nii.gz ${data_data_path}/result/${testing_expname}/fodf.nii.gz ${data_data_path}/result/${testing_expname}/peaks_mrtrix.nii.gz -force
tckgen ${data_data_path}/result/${testing_expname}/fodf.nii.gz ${data_data_path}/result/${testing_expname}/fiber.tck -mask ${data_data_path}/mask.nii.gz -seed_image ${data_data_path}/mask.nii.gz -select 100k -minlength 60 -force
python validation_scripts/tractometer/compute_peak_metric.py --config $1
