all:
	echo "Building permuto_sdf"
	python3 -m pip install -v --user --editable ./ 

clean:
	python3 -m pip uninstall permuto_sdf
	rm -rf build *.egg-info build permuto_sdf*.so libpermuto_sdf_cpp.so libpermuto_sdf_cu.so

        


        
        