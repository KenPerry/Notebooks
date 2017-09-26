#configuration
CONFIG +=  shared qpa no_mocdepend release qt_no_framework
host_build {
    QT_ARCH = x86_64
    QT_TARGET_ARCH = x86_64
} else {
    QT_ARCH = x86_64
    QMAKE_DEFAULT_LIBDIRS = /lib /usr/lib /opt/anaconda1anaconda2anaconda3/lib /opt/anaconda1anaconda2anaconda3/lib/gcc/x86_64-unknown-linux-gnu/4.8.5 /opt/anaconda1anaconda2anaconda3/lib/gcc
    QMAKE_DEFAULT_INCDIRS = /opt/anaconda1anaconda2anaconda3/lib/gcc/x86_64-unknown-linux-gnu/4.8.5/include /opt/anaconda1anaconda2anaconda3/lib/gcc/x86_64-unknown-linux-gnu/4.8.5/include-fixed /opt/anaconda1anaconda2anaconda3/gcc/include/c++ /opt/anaconda1anaconda2anaconda3/gcc/include/c++/x86_64-unknown-linux-gnu /opt/anaconda1anaconda2anaconda3/gcc/include/c++/backward /opt/anaconda1anaconda2anaconda3/include /usr/include
}
QT_CONFIG +=  minimal-config small-config medium-config large-config full-config fontconfig evdev xlib xrender xcb-plugin xcb-qt xcb-glx xcb-sm xkbcommon-qt accessibility-atspi-bridge c++11 accessibility opengl shared qpa reduce_exports reduce_relocations clock-gettime clock-monotonic posix_fallocate mremap getaddrinfo ipv6ifname getifaddrs inotify system-jpeg system-png png system-freetype harfbuzz system-zlib glib dbus dbus-linked openssl xcb rpath alsa gstreamer-1.0 icu concurrent audio-backend release

#versioning
QT_VERSION = 5.6.2
QT_MAJOR_VERSION = 5
QT_MINOR_VERSION = 6
QT_PATCH_VERSION = 2

#namespaces
QT_LIBINFIX = 
QT_NAMESPACE = 

QT_EDITION = OpenSource

QT_COMPILER_STDCXX = 199711
QT_GCC_MAJOR_VERSION = 4
QT_GCC_MINOR_VERSION = 8
QT_GCC_PATCH_VERSION = 5
