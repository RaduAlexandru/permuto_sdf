#include "permuto_sdf/TrainParams.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


TrainParams::TrainParams(const std::string config_file){

    init_params(config_file);
}

void TrainParams::init_params(const std::string config_file){

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);

    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);

    Config train_config=cfg["train"];
    // m_dataset_name=(std::string)train_config["dataset_name"];
    // m_with_viewer = train_config["with_viewer"];
    m_with_visdom = train_config["with_visdom"];
    m_with_tensorboard = train_config["with_tensorboard"];
    m_with_wandb = train_config["with_wandb"];
    // m_lr = train_config["lr"];
    m_save_checkpoint=train_config["save_checkpoint"];

    // if(m_save_checkpoint && !fs::is_directory(m_checkpoint_path)) {
    //     LOG(FATAL) << "The directory for saving checkpoint was not created under " << m_checkpoint_path << ". Maybe you need to create it or maybe you are on the wrong machine.";
    // }

}


// bool TrainParams::with_viewer(){
//     return m_with_viewer;
// }
bool TrainParams::with_visdom(){
    return m_with_visdom;
}
bool TrainParams::with_tensorboard(){
    return m_with_tensorboard;
}
bool TrainParams::with_wandb(){
    return m_with_wandb;
}
// float TrainParams::lr(){
//     return m_lr;
// }
bool TrainParams::save_checkpoint(){
    return m_save_checkpoint;
}
// std::string TrainParams::checkpoint_path(){
//     return m_checkpoint_path;
// }

void TrainParams::set_with_visdom(const bool val){
    m_with_visdom=val;
}
void TrainParams::set_with_tensorboard(const bool val){
    m_with_tensorboard=val;
}
void TrainParams::set_with_wandb(const bool val){
    m_with_wandb=val;
}
void TrainParams::set_save_checkpoint(const bool val){
    m_save_checkpoint=val;
}



