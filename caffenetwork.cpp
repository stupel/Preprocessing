#include "caffenetwork.h"

CaffeNetwork::CaffeNetwork()
{
    this->networkLoaded = false;
}

CaffeNetwork::~CaffeNetwork()
{
    if (this->networkLoaded) {
        net_.reset();
        mean_.release();
    }
}

void CaffeNetwork::run()
{
    exec();
}

std::vector<Prediction> CaffeNetwork::classify(const cv::Mat img)
{
    int N = 5;
    std::vector<float> output = this->predict(img);

    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = caffeArgmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}

void CaffeNetwork::loadModel(const QString &model_file,
                             const QString &trained_file,
                             const QString &mean_file,
                             const QString &label_file)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file.toStdString(), TEST));
    net_->CopyTrainedLayersFrom(trained_file.toStdString());

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels = input_layer->channels();
    CHECK(num_channels == 3 || num_channels == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    this->setMean(mean_file.toStdString());

    /* Load labels. */
    std::ifstream labels(label_file.toStdString().c_str());
    CHECK(labels) << "Unable to open labels file " << label_file.toStdString();
    std::string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels()) << "Number of labels is different from the output layer dimension.";

    this->networkLoaded = true;
}

bool CaffeNetwork::getNetworkLoaded() const
{
    return networkLoaded;
}

void CaffeNetwork::setMean(const std::string& mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels)
            << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> CaffeNetwork::predict(const cv::Mat& img)
{
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels, input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */

    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    this->wrapInputLayer(&input_channels);

    this->preprocess(img, &input_channels);

    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    net_->Forward();

    /* Copy the output layer to a std::vector */
    output_layer = net_->output_blobs()[0];
    begin = output_layer->cpu_data();
    end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

void CaffeNetwork::wrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void CaffeNetwork::preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;

    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    sample_normalized /= 255.0;
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

void CaffeNetwork::preprocessBatch(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch)
{
    for (int i = 0 ; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

        /* Convert the input image to the input image format of the network. */
        cv::Mat sample;
        if (img.channels() == 3 && this->num_channels == 1)
          cv::cvtColor(img, sample, CV_BGR2GRAY);
        else if (img.channels() == 4 && this->num_channels == 1)
          cv::cvtColor(img, sample, CV_BGRA2GRAY);
        else if (img.channels() == 4 && this->num_channels == 3)
          cv::cvtColor(img, sample, CV_BGRA2BGR);
        else if (img.channels() == 1 && this->num_channels == 3)
          cv::cvtColor(img, sample, CV_GRAY2BGR);
        else
          sample = img;

        cv::Mat sample_resized;
        if (sample.size() != this->input_geometry_)
          cv::resize(sample, sample_resized, this->input_geometry_);
        else
          sample_resized = sample;

        cv::Mat sample_float;
        if (this->num_channels == 3)
          sample_resized.convertTo(sample_float, CV_32FC3);
        else
          sample_resized.convertTo(sample_float, CV_32FC1);

        cv::Mat sample_normalized;

        cv::subtract(sample_float, this->mean_, sample_normalized);

        /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
        cv::split(sample_normalized, *input_channels);

//        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//              == net_->input_blobs()[0]->cpu_data())
//          << "Input channels are not wrapping the input layer of the network.";
    }
}

void CaffeNetwork::wrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch)
{
    Blob<float>* input_layer_ = this->net_->input_blobs()[0];

    int width = input_layer_->width();
    int height = input_layer_->height();
    int num = input_layer_->num();
    float* input_data = input_layer_->mutable_cpu_data();
    for ( int j = 0; j < num; j++){
        vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer_->channels(); ++i){
          cv::Mat channel(height, width, CV_32FC1, input_data);
          input_channels.push_back(channel);
          input_data += width * height;
        }
        input_batch -> push_back(vector<cv::Mat>(input_channels));
    }
}

std::vector<float> CaffeNetwork::predictBatch(const vector< cv::Mat > imgs)
{
    Blob<float>* input_layer;
    if (!this->networkLoaded) input_layer = this->net_->input_blobs()[0];
    input_layer->Reshape(imgs.size(), this->num_channels,
                         this->input_geometry_.height,
                         this->input_geometry_.width);

    /* Forward dimension change to all layers. */
    this->net_->Reshape();

    std::vector< std::vector<cv::Mat> > input_batch;
    this->wrapBatchInputLayer(&input_batch);

    this->preprocessBatch(imgs, &input_batch);

    this->net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = this->net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels()*imgs.size();

    delete input_layer;

    return std::vector<float>(begin, end);
}

std::vector<vector<Prediction>> CaffeNetwork::classifyBatch(const vector< cv::Mat > imgs, int num_classes)
{
    Caffe::set_mode(Caffe::GPU);

    std::vector<float> output_batch = predictBatch(imgs);

    std::vector< std::vector<Prediction> > predictions;
    for(int j = 0; j < imgs.size(); j++){
        std::vector<float> output(output_batch.begin() + j*num_classes, output_batch.begin() + (j+1)*num_classes);
        std::vector<int> maxN = caffeArgmax(output, num_classes);
        std::vector<Prediction> prediction_single;
        for (int i = 0; i < num_classes; ++i) {
          int idx = maxN[i];
          prediction_single.push_back(std::make_pair(this->labels_[idx], output[idx]));
        }
        predictions.push_back(std::vector<Prediction>(prediction_single));
    }
    return predictions;
}

void CaffeNetwork::convertImageset(QVector<QPair<QString, int>> lines, const QString &outputDir, QString setName)
{
    emit logSignal("trainer", "Starting converting the imageset...");
    emit updateProgressBarSignal("trainer", 0, "Converting " + setName + " imageset");

    //::google::InitGoogleLogging(outputDir);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

    namespace gflags = google;

    bool is_color = true;                     //When this option is on, treat images as grayscale ones
    bool check_size = false;                  //When this option is on, check that all the datum have the same size
    bool encoded = false;                     //When this option is on, the encoded image will be save in datum
    string encode_type = "";                  //Optional: What type should we encode the image as ('png','jpg',...)
    bool shuffle = false;                     //Randomly shuffle the order of images and their labels
    int resize_width = 0;                     //Width images are resized to
    int resize_height = 0;                    //Height images are resized to
    string backend = "lmdb";


    // Check for folder exists
    if (QDir(outputDir + "/frequency_" + setName + "-lmdb").exists()) {
        emit logSignal("trainer", "Folder frequency_" + setName + "-lmdb exists, removing...");
        QDir(outputDir + "/frequency_" + setName + "-lmdb").removeRecursively();
        emit logSignal("trainer", "Removed");
    }

    if (shuffle) {
        QPair<QString, int> item;
        int randItem;
        qsrand(QTime::currentTime().msec());
        for (int i = 0; i < lines.size(); i++) {
            randItem = qrand() % lines.size();
            item = lines[randItem];
            lines.removeAt(randItem);
            lines.push_back(item);
        }
        emit logSignal("trainer", "Shuffling data");
    }
    emit logSignal("trainer", "A total of " + QString::number(lines.size()) + " images.");

    if (encode_type.size() && !encoded)
        emit logSignal("trainer", "- encode_type specified, assuming encoded=true.");

    // Create new DB
    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(outputDir.toStdString() + "/frequency_" + setName.toStdString() + "-lmdb", db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    Datum datum;
    int count = 0;
    int data_size = 0;
    bool data_size_initialized = false;

    for (int line_id = 0; line_id < lines.size(); ++line_id) {
        bool status;
        std::string enc = encode_type;
        if (encoded && !enc.size()) {
            // Guess the encoding type from the file name
            string fn = lines[line_id].first.toStdString();
            size_t p = fn.rfind('.');
            if ( p == fn.npos )
                emit logSignal("trainer", "Failed to guess the encoding of '" + QString::fromStdString(fn) + "'");
            enc = fn.substr(p);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        status = ReadImageToDatum(lines[line_id].first.toStdString(),
                                  lines[line_id].second, resize_height, resize_width, is_color,
                                  enc, &datum);
        if (status == false) continue;
        if (check_size) {
            if (!data_size_initialized) {
                data_size = datum.channels() * datum.height() * datum.width();
                data_size_initialized = true;
            } else {
                const std::string& data = datum.data();
                CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                                                 << data.size();
            }
        }
        // sequential
        string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first.toStdString();

        // Put in db
        string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str, out);

        if (++count % 1000 == 0) {
            // Commit db
            txn->Commit();
            txn.reset(db->NewTransaction());
            emit logSignal("trainer", "Processed " + QString::number(count) + " files.");
            emit updateProgressBarSignal("trainer", (int)(count*1.0/lines.size() * 100), "Converting " + setName + " imageset");
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
        txn->Commit();
        emit logSignal("trainer", "Processed " + QString::number(count) + " files.");
        emit updateProgressBarSignal("trainer", 100, "Converting " + setName + " imageset");
    }

    this->computeImageMean(outputDir, setName, count);
}

void CaffeNetwork::computeImageMean(const QString &outputDir, QString setName, int totalCount)
{
    emit logSignal("trainer", "Computing image mean...");
    emit updateProgressBarSignal("trainer", 0, "Computing " + setName + " imagemean");

    //::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

    namespace gflags = google;

    string backend = "lmdb";     // The backend {leveldb, lmdb} containing the images

    scoped_ptr<db::DB> db(db::GetDB(backend));
    db->Open(outputDir.toStdString() + "/frequency_" + setName.toStdString() + "-lmdb" , db::READ);
    scoped_ptr<db::Cursor> cursor(db->NewCursor());

    BlobProto sum_blob;
    int count = 0;
    // load first datum
    Datum datum;
    datum.ParseFromString(cursor->value());

    if (DecodeDatumNative(&datum)) {
        emit logSignal("trainer", "Decoding Datum");
    }

    sum_blob.set_num(1);
    sum_blob.set_channels(datum.channels());
    sum_blob.set_height(datum.height());
    sum_blob.set_width(datum.width());
    const int data_size = datum.channels() * datum.height() * datum.width();
    int size_in_datum = std::max<int>(datum.data().size(),
                                      datum.float_data_size());
    for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.add_data(0.);
    }
    emit logSignal("trainer", "Starting iteration");
    while (cursor->valid()) {
        Datum datum;
        datum.ParseFromString(cursor->value());
        DecodeDatumNative(&datum);

        const std::string& data = datum.data();
        size_in_datum = std::max<int>(datum.data().size(),
                                      datum.float_data_size());
        CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
                                              size_in_datum;
        if (data.size() != 0) {
            CHECK_EQ(data.size(), size_in_datum);
            for (int i = 0; i < size_in_datum; ++i) {
                sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
            }
        } else {
            CHECK_EQ(datum.float_data_size(), size_in_datum);
            for (int i = 0; i < size_in_datum; ++i) {
                sum_blob.set_data(i, sum_blob.data(i) +
                                  static_cast<float>(datum.float_data(i)));
            }
        }
        ++count;
        if (count % 10000 == 0) {
            emit logSignal("trainer", "Processed " + QString::number(count) + " files.");
            emit updateProgressBarSignal("trainer", (int)(count*1.0/totalCount * 100), "Computing " + setName + " imagemean");
        }
        cursor->Next();
    }

    if (count % 10000 != 0) {
        emit logSignal("trainer", "Processed " + QString::number(count) + " files.");
        emit updateProgressBarSignal("trainer", (int)(count*1.0/totalCount * 100), "Computing " + setName + " imagemean");
    }
    for (int i = 0; i < sum_blob.data_size(); ++i) {
        sum_blob.set_data(i, sum_blob.data(i) / count);
    }

    emit logSignal("trainer", "Write to " + setName + "-imagemean.binaryproto");
    WriteProtoToBinaryFile(sum_blob, outputDir.toStdString() + "/frequency_" + setName.toStdString() + "-imagemean.binaryproto");

    const int channels = sum_blob.channels();
    const int dim = sum_blob.height() * sum_blob.width();
    std::vector<float> mean_values(channels, 0.0);
    emit logSignal("trainer", "Number of channels: " + QString::number(channels));
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < dim; ++i) {
            mean_values[c] += sum_blob.data(dim * c + i);
        }
        emit logSignal("trainer", "mean_value channel [" + QString::number(c) + "]: " + QString::number(mean_values[c] / dim));
    }

    if (setName == "test") this->trainNetworkWithExe(outputDir);
    //if (setName == "test") this->trainNetwork();
}

void CaffeNetwork::trainNetworkWithExe(const QString &outputDir)
{
    emit logSignal("trainer", "Starting training process...");
    emit updateProgressBarSignal("trainer", 0, "Training");

    QProcess *caffeExe = new QProcess();
    QString cmd;

    cmd = QString("\"./core/extern/caffe.exe\" ") + "train -solver ./core/config/Caffe/frequency_solver.prototxt";
    qDebug() << cmd;

    caffeExe->execute(cmd);
    caffeExe->waitForFinished();

    emit trainingDoneSignal();

    delete caffeExe;

    emit logSignal("trainer", "Training Done");
}

void CaffeNetwork::trainNetwork()
{
    emit logSignal("trainer", "Starting training process...");
    string flag_gpu = "all";          //Optional; run in GPU mode on given device IDs separated by ','.
                                    //Use '-gpu all' to run on all available GPUs. The effective training
                                    //batch size is multiplied by the number of devices.);
    string flag_solver = "./core/config/Caffe/frequency_solver.prototxt";       //The solver definition protocol buffer text file.
    string flag_snapshot = "10";     //Optional; the snapshot solver state to resume training.
    int flag_level = 0;            //Optional; network level.
    string flag_weights = "";      //Optional; the pretrained weights to initialize finetuning,
                                    //separated by ','. Cannot be set simultaneously with snapshot.
    string flag_sigint_effect = "stop";        //Optional; action to take when a SIGINT signal is received: snapshot, stop or none
    string flag_sighup_effect = "snapshot";    //Optional; action to take when a SIGHUP signal is received: snapshot, stop or none

    const char *modelPath = "./core/config/Caffe/frequency_model.prototxt";


    CHECK_GT(flag_solver.size(), 0) << "Need a solver definition to train.";
    CHECK(!flag_snapshot.size() || !flag_weights.size())
            << "Give a snapshot to resume training or weights to finetune "
               "but not both.";

    vector<string> stages = get_stages_from_flags();            //!!!!!!!!!!!!!

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(flag_solver, &solver_param);
    //solver_param.set_solver_mode(caffe::SolverParameter_SolverMode_GPU);
    //solver_param.set_net(modelPath, sizeof(modelPath));

    solver_param.mutable_train_state()->set_level(flag_level);
    for (int i = 0; i < stages.size(); i++) {
        solver_param.mutable_train_state()->add_stage(stages[i]);
    }

    // If the gpus flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
    if (flag_gpu.size() == 0
            && solver_param.has_solver_mode()
            && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
        if (solver_param.has_device_id()) {
            flag_gpu = "" + boost::lexical_cast<std::string>(solver_param.device_id());
        } else {  // Set default GPU if unspecified
            flag_gpu = "" + boost::lexical_cast<std::string>(0);
        }
    }

    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() == 0) {
        emit logSignal("trainer", "Use CPU.");
        Caffe::set_mode(Caffe::CPU);
    } else {
        ostringstream s;
        for (int i = 0; i < gpus.size(); ++i) {
            s << (i ? ", " : "") << gpus[i];
        }
        emit logSignal("trainer", "Using GPUs");

#ifndef CPU_ONLY
        cudaDeviceProp device_prop;
        for (int i = 0; i < gpus.size(); ++i) {
            cudaGetDeviceProperties(&device_prop, gpus[i]);
            emit logSignal("trainer", "GPU " + QString::number(gpus[i]) + ": " + device_prop.name);
        }
#endif
        solver_param.set_device_id(gpus[0]);
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(gpus.size());
    }

    caffe::SignalHandler signal_handler(
                GetRequestedAction(flag_sigint_effect),
                GetRequestedAction(flag_sighup_effect));

    shared_ptr<caffe::Solver<float> >
            solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    solver->SetActionFunction(signal_handler.GetActionFunction());

    if (flag_snapshot.size()) {
        emit logSignal("trainer", "Resuming from " + QString::fromStdString(flag_snapshot));
        solver->Restore(flag_snapshot.c_str());
    } else if (flag_weights.size()) {
        CopyLayers(solver.get(), flag_weights);
    }

    emit logSignal("trainer", "Starting Optimization");

    if (gpus.size() > 1) {
#ifdef USE_NCCL
        caffe::NCCL<float> nccl(solver);
        nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
        emit logSignal("trainer", "Multi-GPU execution not available - rebuild with USE_NCCL");
#endif
    } else {
        solver->Solve();
    }
    emit logSignal("trainer", "Optimization Done");
}

// Parse GPU ids or use all available devices
void CaffeNetwork::get_gpus(vector<int>* gpus)
{
    string flag_gpu = "";

    if (flag_gpu == "all") {
        int count = 0;
#ifndef CPU_ONLY
        CUDA_CHECK(cudaGetDeviceCount(&count));
#else
        NO_GPU;
#endif
        for (int i = 0; i < count; ++i) {
            gpus->push_back(i);
        }
    } else if (flag_gpu.size()) {
        vector<string> strings;
        boost::split(strings, flag_gpu, boost::is_any_of(","));
        for (int i = 0; i < strings.size(); ++i) {
            gpus->push_back(boost::lexical_cast<int>(strings[i]));
        }
    } else {
        CHECK_EQ(gpus->size(), 0);
    }
}

vector<string> CaffeNetwork::get_stages_from_flags()
{
    string flag_stage = "";

    vector<string> stages;
    boost::split(stages, flag_stage, boost::is_any_of(","));
    return stages;
}

caffe::SolverAction::Enum CaffeNetwork::GetRequestedAction(const std::string& flag_value)
{
    if (flag_value == "stop") {
        return caffe::SolverAction::STOP;
    }
    if (flag_value == "snapshot") {
        return caffe::SolverAction::SNAPSHOT;
    }
    if (flag_value == "none") {
        return caffe::SolverAction::NONE;
    }
    emit logSignal("trainer", "Invalid signal effect \"" + QString::fromStdString(flag_value) + "\" was specified");
}

void CaffeNetwork::CopyLayers(caffe::Solver<float>* solver, const std::string& model_list)
{
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(",") );
    for (int i = 0; i < model_names.size(); ++i) {
        emit logSignal("trainer", "Finetuning from " + QString::fromStdString(model_names[i]));
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < solver->test_nets().size(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}
