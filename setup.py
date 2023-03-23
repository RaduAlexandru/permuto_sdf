import os
import re
import sys
import platform
import subprocess
import glob

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils.command.install_headers import install_headers as install_headers_orig
from setuptools import setup
from distutils.sysconfig import get_python_inc
import site #section 2.7 in https://docs.python.org/3/distutils/setupscript.html
# import catkin.workspace


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
                    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable
                    #   '-GNinja'
                      ]

        # cfg = 'Debug' if self.debug else 'Release'
        # build_args = ['--config', cfg]
        build_args = []

        if platform.system() == "Windows":
            # cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            # cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # print ("build temp is ", self.build_temp)

        #find out where do the header file and the shader files get copied into https://stackoverflow.com/questions/14375222/find-python-header-path-from-within-python
        # print("PRINTING-------------------------------------------------")
        # print( get_python_inc() ) #this gives the include dir
        # print( site.USER_BASE )  # data files that are added through data_files=[] get added here for --user instaltions https://docs.python.org/3/distutils/setupscript.html
        # pkg_dir, dist = self.create_dist(headers="dummy_header")
        # dummy_install_headers=install_headers_orig(dist=dist)
        # help(install_headers_orig)
        # dummy_install_headers=install_headers_orig(self.distribution)
        # print( dummy_install_headers.install_dir ) #this gives the include dir
        # cmake_args+=['-DEASYPBR_SHADERS_PATH=' + get_python_inc()+"/easypbr"]
        # cmake_args+=['-DEASYPBR_SHADERS_PATH=' + site.USER_BASE]
        # cmake_args+=['-DDATA_DIR=' + site.USER_BASE]
        # cmake_args+=['-DCATKIN_PACKAGE_LIB_DESTINATION=' + "./"]


        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        # subprocess.check_call(['make', 'install'], cwd=self.build_temp)
        # subprocess.check_call(['catkin build', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        # subprocess.check_call(['catkin', 'build' ,'--this'] + build_args  + cmake_args, cwd=self.build_temp, env=env)

        # ##get the workspace path depending on the current path and catkin.workspace.get_workspaces
        # # print("current path")
        # cur_dir=os.path.dirname(os.path.abspath(__file__))
        # workspaces=catkin.workspace.get_workspaces()
        # # current_workspace=[x if cur_dir in  else '' for x in row]
        # current_workspace=""
        # for path in workspaces:
        #     last_part=os.path.basename(os.path.normpath(path))
        #     if(last_part=="devel"):
        #        potential_workspace=os.path.dirname(path) #gets rid of /devel
        #        if(potential_workspace in cur_dir):
        #            current_workspace=potential_workspace
        #            break
        # print("current workspace is ", current_workspace)

        # #simlink the libraries from the devel space towards the current dir so that the egg link can find them
        # catkin_lib_dir=os.path.join(current_workspace,"devel/lib/")
        # libs = [f for f in os.listdir(catkin_lib_dir) if os.path.isfile(os.path.join(catkin_lib_dir, f))]
        # print(libs)
        # for lib in libs:
        #     if "dataloaders" in lib:
        #         print ("linking ", lib)
        #         print("cmd", ['ln', '-sf'] + [ os.path.join(catkin_lib_dir,lib) + " " + os.path.join(cur_dir, lib ) ] )
        #         subprocess.check_call(['ln', '-sf'] + [ os.path.join(catkin_lib_dir,lib) ] +  [ os.path.join(cur_dir, lib ) ]  )
        #         # subprocess.check_call(['cp'] + [ os.path.join(catkin_lib_dir,lib) + " " + os.path.join(cur_dir, lib )]  )

       

with open("README.md", "r") as fh:
    long_description = fh.read()

#install headers while retaining the structure of the tree folder https://stackoverflow.com/a/50114715
class install_headers(install_headers_orig):

    def run(self):
        headers = self.distribution.headers or []
        for header in headers:
            dst = os.path.join(self.install_dir, os.path.dirname(header))
            print("----------------copying in ", dst)
            # dst = os.path.join(get_python_inc(), os.path.dirname(header))
            self.mkpath(dst)
            (out, _) = self.copy_file(header, dst)
            self.outfiles.append(out)

def has_any_extension(filename, extensions):
    for each in extensions:
        if filename.endswith(each):
            return True
    return False

# https://stackoverflow.com/a/41031566
def find_files(directory, strip, extensions):
    """
    Using glob patterns in ``package_data`` that matches a directory can
    result in setuptools trying to install that directory as a file and
    the installation to fail.

    This function walks over the contents of *directory* and returns a list
    of only filenames found. The filenames will be stripped of the *strip*
    directory part.

    It only takes file that have a certain extension
    """

    result = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # if filename.endswith('.h') or filename.endswith('.hpp') or filename.endswith('.cuh'):
            if has_any_extension(filename, extensions):
                # print("filename", filename)
                filename = os.path.join(root, filename)
                result.append(os.path.relpath(filename, strip))

    # print("result of find_files is ", result)

    return result
    # return 'include/easy_pbr/LabelMngr.h'

setup(
    name='permuto_sdf',
    version='1.0.0',
    author="Radu Alexandru Rosu",
    author_email="rosu@ais.uni-bonn.de",
    description="permuto_sdf",
    long_description=long_description,
    ext_modules=[CMakeExtension('permuto_sdf')],
    cmdclass={ 'build_ext':CMakeBuild,
               'install_headers': install_headers,
    },
    zip_safe=False,
)



