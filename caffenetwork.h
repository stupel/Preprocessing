#ifndef CAFFENETWORK_H
#define CAFFENETWORK_H
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <QObject>
#include <QThread>
#include <QDebug>
#include <QTime>
#include <QDir>
#include <QProcess>

//Caffe
#include <caffe/caffe.hpp>

//OpenCV
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/signal_handler.h"

using std::pair;
using boost::scoped_ptr;

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

using namespace caffe;

static bool caffePairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> caffeArgmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), caffePairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class CaffeNetwork : public QThread
{
    Q_OBJECT

public:
    CaffeNetwork();
    ~CaffeNetwork();

    std::vector<Prediction> classify(const cv::Mat img);
    std::vector<vector<Prediction>> classifyBatch(const vector<cv::Mat> imgs, int num_classes);

    bool getNetworkLoaded() const;

public slots:
    void loadModel(const QString& model_file,
                   const QString& trained_file,
                   const QString& mean_file,
                   const QString& label_file);
    void convertImageset(QVector<QPair<QString, int> > lines, const QString &outputDir, QString setName);

private:
    QObject *parent;

    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels;
    cv::Mat mean_;
    std::vector<std::string> labels_;
    int batchSize;

    bool networkLoaded;

    void run();

    void setMean(const std::string& mean_file);
    std::vector<float> predict(const cv::Mat& img);
    std::vector<float> predictBatch(const vector< cv::Mat > imgs);
    void wrapInputLayer(std::vector<cv::Mat>* input_channels);
    void wrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);
    void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    void preprocessBatch(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch);
    void computeImageMean(const QString &outputDir, QString setName, int totalCount);
    void trainNetwork();
    void trainNetworkWithExe(const QString &outputDir);

    static void get_gpus(vector<int>* gpus);
    vector<string> get_stages_from_flags();
    caffe::SolverAction::Enum GetRequestedAction(const std::string& flag_value);
    void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list);

signals:
    void logSignal(QString, QString);
    void updateProgressBarSignal(QString, int, QString);
    void trainingDoneSignal();
};

#endif // CAFFENETWORK_H
