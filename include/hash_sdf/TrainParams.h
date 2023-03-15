#pragma once 

#include <memory>
#include <stdarg.h>

//class used to read some network training parameters from a config file ( things like learnign rate and batch size ) This class is also exposed to python so it can be used in pytorch

class TrainParams: public std::enable_shared_from_this<TrainParams>
{
public:
    template <class ...Args>
    static std::shared_ptr<TrainParams> create( Args&& ...args ){
        return std::shared_ptr<TrainParams>( new TrainParams(std::forward<Args>(args)...) );
    }

    // bool with_viewer();
    bool with_visdom();
    bool with_tensorboard();
    bool with_wandb();
    bool save_checkpoint();

    //setters
    void set_with_visdom(const bool val);
    void set_with_tensorboard(const bool val);
    void set_with_wandb(const bool val);
    void set_save_checkpoint(const bool val);


private:
    TrainParams(const std::string config_file);
    void init_params(const std::string config_file);

    // bool m_with_viewer;
    bool m_with_visdom;
    bool m_with_tensorboard;
    bool m_with_wandb;
    // float m_lr; 
    bool m_save_checkpoint;

};