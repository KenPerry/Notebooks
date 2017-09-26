CONFIG +=  compile_examples qpa largefile precompile_header enable_new_dtags sse2 sse3 ssse3 sse4_1 sse4_2 pcre
QT_BUILD_PARTS += libs tools
QT_SKIP_MODULES +=  qtenginio qtlocation qtsensors qtserialport qtserialbus qtquickcontrols2 qtwayland qtcanvas3d qt3d qtwebengine
QT_NO_DEFINES =  CUPS EGL EGLFS EGL_X11 EVENTFD ICONV IMAGEFORMAT_JPEG LIBPROXY OPENVG PULSEAUDIO STYLE_GTK TSLIB ZLIB
QT_QCONFIG_PATH = 
host_build {
    QT_CPU_FEATURES.x86_64 =  mmx sse sse2
} else {
    QT_CPU_FEATURES.x86_64 =  mmx sse sse2
}
QT_COORD_TYPE = double
QT_LFLAGS_ODBC   = -lodbc
styles += mac fusion windows
DEFINES += QT_NO_MTDEV
QT_LIBS_DBUS = -L/opt/anaconda1anaconda2anaconda3/lib -ldbus-1  
QT_CFLAGS_DBUS = -I/opt/anaconda1anaconda2anaconda3/include/dbus-1.0 -I/opt/anaconda1anaconda2anaconda3/lib/dbus-1.0/include  
QT_HOST_CFLAGS_DBUS = -I/opt/anaconda1anaconda2anaconda3/include/dbus-1.0 -I/opt/anaconda1anaconda2anaconda3/lib/dbus-1.0/include  
QT_CFLAGS_GLIB = -pthread -I/opt/anaconda1anaconda2anaconda3/include/glib-2.0 -I/opt/anaconda1anaconda2anaconda3/lib/glib-2.0/include -I/opt/anaconda1anaconda2anaconda3/include  
QT_LIBS_GLIB = -pthread -L/opt/anaconda1anaconda2anaconda3/lib -lgthread-2.0 -lrt -lglib-2.0  
QMAKE_CFLAGS_FONTCONFIG = -I/opt/anaconda1anaconda2anaconda3/include -I/opt/anaconda1anaconda2anaconda3/include/freetype2 -I/opt/anaconda1anaconda2anaconda3/include/libxml2  
QMAKE_LIBS_FONTCONFIG = -L/opt/anaconda1anaconda2anaconda3/lib -lfontconfig -lfreetype  
DEFINES += QT_NO_LIBUDEV
DEFINES += QT_NO_TSLIB
DEFINES += QT_NO_LIBINPUT
QMAKE_X11_PREFIX = /usr
QMAKE_XKB_CONFIG_ROOT = /opt/anaconda1anaconda2anaconda3/lib
EXTRA_DEFINES +=  "_X_INLINE=inline" "XK_dead_currency=0xfe6f" "XK_ISO_Level5_Lock=0xfe13" "FC_WEIGHT_EXTRABLACK=215" "FC_WEIGHT_ULTRABLACK=FC_WEIGHT_EXTRABLACK" "GLX_GLXEXT_PROTOTYPES"
EXTRA_INCLUDEPATH +=  "/opt/anaconda1anaconda2anaconda3/include"
EXTRA_LIBS +=  -L"/opt/anaconda1anaconda2anaconda3/lib"
sql-drivers = 
sql-plugins =  sqlite
