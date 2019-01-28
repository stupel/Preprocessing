QT       += gui

Debug:TARGET = Preprocessingd
Release:TARGET = Preprocessing
TEMPLATE = lib

DEFINES += PREPROCESSING_LIBRARY
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += PRO_PWD=\\\"$$_PRO_FILE_PWD_\\\"

QMAKE_CFLAGS_ISYSTEM=

SOURCES += preprocessing.cpp \
    orientationmap.cpp \
    thinning.cpp \
    binarization.cpp \
    gaborfiltermultithread.cpp \
    gaborthread.cpp \
    contrastenhancement.cpp \
    frequencymap.cpp \
    mask.cpp \
    qualitymap.cpp \
    gaborfiltergpu.cpp \
    preprocessing_caffenetwork.cpp

HEADERS += preprocessing.h\
        preprocessing_global.h \
    helper.h \
    orientationmap.h \
    thinning.h \
    binarization.h \
    gaborfiltermultithread.h \
    gaborthread.h \
    contrastenhancement.h \
    frequencymap.h \
    mask.h \
    imagecontour.h \
    qualitymap.h \
    gaborfiltergpu.h \
    preprocessing_config.h \
    preprocessing_caffenetwork.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

#CUDA
unix:!macx: LIBS += -L$$PWD/../../../../../opt/cuda/lib64/ -lcudart
INCLUDEPATH += $$PWD/../../../../../opt/cuda/include
DEPENDPATH += $$PWD/../../../../../opt/cuda/include

#ArrayFire
unix:!macx: LIBS += -L$$PWD/../../../../../usr/lib/ -lafcuda

#OpenCV
INCLUDEPATH += /usr/include/opencv4
