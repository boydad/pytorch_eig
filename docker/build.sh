#!/bin/bash


#tag=22.02
#tag=debug
#tag=jaideep_hpo2
#tag=tkurth_hpo2
#tag=andrea
name=pyeig
dlfw_version=22.03
tag=${dlfw_version}
cd ../

# build
docker build -t ${name}:${tag} --build-arg DLFW_VERSION=${dlfw_version} -f docker/Dockerfile .

# push
#docker push ${repo}:${tag}

# retag and repush
#docker tag ${repo}:${tag} thorstenkurth/era5-wind:${tag}
#docker push thorstenkurth/era5-wind:${tag}
