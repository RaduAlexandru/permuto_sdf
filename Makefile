all:
	echo "Building hash_sdf"
	python3 -m pip install -v --user --editable ./ 

clean:
	python3 -m pip uninstall hash_sdf
	rm -rf build *.egg-info build hash_sdf*.so libhash_sdf_cpp.so libhash_sdf_cu.so

        


        
        