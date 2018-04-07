QT       += gui

Debug:TARGET = Preprocessingd
Release:TARGET = Preprocessing
TEMPLATE = lib

DEFINES += PREPROCESSING_LIBRARY
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += PRO_PWD=\\\"$$_PRO_FILE_PWD_\\\"

SOURCES += preprocessing.cpp \
    orientationmap.cpp \
    thinning.cpp \
    binarization.cpp \
    gaborfiltermultithread.cpp \
    gaborthread.cpp \
    contrastenhancement.cpp \
    frequencymap.cpp \
    caffenetwork.cpp \
    mask.cpp \
    qualitymap.cpp

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
    caffenetwork.h \
    mask.h \
    config.h \
    imagecontour.h \
    qualitymap.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}


unix:!macx: LIBS += -L$$PWD/../../../../../opt/cuda/lib64/ -lcudart

INCLUDEPATH += $$PWD/../../../../../opt/cuda/include
DEPENDPATH += $$PWD/../../../../../opt/cuda/include
