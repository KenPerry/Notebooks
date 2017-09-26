# (c) 2012-2016 Continuum Analytics, Inc. / http://continuum.io
# All Rights Reserved
#
# conda is distributed under the terms of the BSD 3-clause license.
# Consult LICENSE.txt or http://opensource.org/licenses/BSD-3-Clause.
'''
We use the following conventions in this module:

    dist:        canonical package name, e.g. 'numpy-1.6.2-py26_0'

    ROOT_PREFIX: the prefix to the root environment, e.g. /opt/anaconda

    PKGS_DIR:    the "package cache directory", e.g. '/opt/anaconda/pkgs'
                 this is always equal to ROOT_PREFIX/pkgs

    prefix:      the prefix of a particular environment, which may also
                 be the root environment

Also, this module is directly invoked by the (self extracting) tarball
installer to create the initial environment, therefore it needs to be
standalone, i.e. not import any other parts of `conda` (only depend on
the standard library).
'''
import os
import re
import sys
import json
import shutil
import stat
from os.path import abspath, dirname, exists, isdir, isfile, islink, join
from optparse import OptionParser


on_win = bool(sys.platform == 'win32')
try:
    FORCE = bool(int(os.getenv('FORCE', 0)))
except ValueError:
    FORCE = False

LINK_HARD = 1
LINK_SOFT = 2  # never used during the install process
LINK_COPY = 3
link_name_map = {
    LINK_HARD: 'hard-link',
    LINK_SOFT: 'soft-link',
    LINK_COPY: 'copy',
}
SPECIAL_ASCII = '$!&\%^|{}[]<>~`"\':;?@*#'

# these may be changed in main()
ROOT_PREFIX = sys.prefix
PKGS_DIR = join(ROOT_PREFIX, 'pkgs')
SKIP_SCRIPTS = False
IDISTS = {
  "_license-1.1-py27_1": {
    "md5": "dda038d00d891fc1b747471b2d248595", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/_license-1.1-py27_1.tar.bz2"
  }, 
  "alabaster-0.7.10-py27_0": {
    "md5": "598e3584c608e13332b3ee56f1a43b38", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/alabaster-0.7.10-py27_0.tar.bz2"
  }, 
  "anaconda-4.4.0-np112py27_0": {
    "md5": "109cb6b00e257f7211b61d0e0b3736e2", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/anaconda-4.4.0-np112py27_0.tar.bz2"
  }, 
  "anaconda-client-1.6.3-py27_0": {
    "md5": "44a1a7421ef0a1b6da573654edc984fe", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/anaconda-client-1.6.3-py27_0.tar.bz2"
  }, 
  "anaconda-navigator-1.6.2-py27_0": {
    "md5": "10cd113844da87ba14c450a313649ab1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/anaconda-navigator-1.6.2-py27_0.tar.bz2"
  }, 
  "anaconda-project-0.6.0-py27_0": {
    "md5": "6dceba1617e3718971bce528e0cf37e6", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/anaconda-project-0.6.0-py27_0.tar.bz2"
  }, 
  "asn1crypto-0.22.0-py27_0": {
    "md5": "043e31b65654f60b9a487043c632ea3e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/asn1crypto-0.22.0-py27_0.tar.bz2"
  }, 
  "astroid-1.4.9-py27_0": {
    "md5": "cf0e2013b32ecb5465efc9086ce67271", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/astroid-1.4.9-py27_0.tar.bz2"
  }, 
  "astropy-1.3.2-np112py27_0": {
    "md5": "4f242fd087927e5aed720892faac0508", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/astropy-1.3.2-np112py27_0.tar.bz2"
  }, 
  "babel-2.4.0-py27_0": {
    "md5": "8c3017f4d1ab4917c73383c2ac40f948", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/babel-2.4.0-py27_0.tar.bz2"
  }, 
  "backports-1.0-py27_0": {
    "md5": "3584e3c89857146300d90018dcd70c10", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/backports-1.0-py27_0.tar.bz2"
  }, 
  "backports_abc-0.5-py27_0": {
    "md5": "ccee4e39eb674c2fc92edbdbe4431820", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/backports_abc-0.5-py27_0.tar.bz2"
  }, 
  "beautifulsoup4-4.6.0-py27_0": {
    "md5": "ac0001adfc9ae63978716985574ae7c3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/beautifulsoup4-4.6.0-py27_0.tar.bz2"
  }, 
  "bitarray-0.8.1-py27_0": {
    "md5": "577cd6c105d2ec1c2bbfba29ddc7a88a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/bitarray-0.8.1-py27_0.tar.bz2"
  }, 
  "blaze-0.10.1-py27_0": {
    "md5": "76667d843b5f94468af1c5e7300b2684", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/blaze-0.10.1-py27_0.tar.bz2"
  }, 
  "bleach-1.5.0-py27_0": {
    "md5": "38fdcdf255c8f4e3763406531939200a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/bleach-1.5.0-py27_0.tar.bz2"
  }, 
  "bokeh-0.12.5-py27_1": {
    "md5": "d3e8f9d9ce40f2be2ce64378673f0674", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/bokeh-0.12.5-py27_1.tar.bz2"
  }, 
  "boto-2.46.1-py27_0": {
    "md5": "7e230943ed6d67c2382a735111e7bf78", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/boto-2.46.1-py27_0.tar.bz2"
  }, 
  "bottleneck-1.2.1-np112py27_0": {
    "md5": "3ed7611a36987c2e7b4d1703a66746be", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/bottleneck-1.2.1-np112py27_0.tar.bz2"
  }, 
  "cairo-1.14.8-0": {
    "md5": "c373c512434e418cdfee4e6cd9f0a294", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cairo-1.14.8-0.tar.bz2"
  }, 
  "cdecimal-2.3-py27_2": {
    "md5": "211b163663e5f4ced130f0f950c813e9", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cdecimal-2.3-py27_2.tar.bz2"
  }, 
  "cffi-1.10.0-py27_0": {
    "md5": "ad28ba19cf4ee2c19c468ab6a9725f89", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cffi-1.10.0-py27_0.tar.bz2"
  }, 
  "chardet-3.0.3-py27_0": {
    "md5": "6cfb294a6c3045fb595637c7b4ab311d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/chardet-3.0.3-py27_0.tar.bz2"
  }, 
  "click-6.7-py27_0": {
    "md5": "b06ea52ac2f180651ca2b066cce7f1b7", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/click-6.7-py27_0.tar.bz2"
  }, 
  "cloudpickle-0.2.2-py27_0": {
    "md5": "97934ebb7ccbb40193b64813f4728bf8", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cloudpickle-0.2.2-py27_0.tar.bz2"
  }, 
  "clyent-1.2.2-py27_0": {
    "md5": "97e9268eabcb5ccded58fb9188730786", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/clyent-1.2.2-py27_0.tar.bz2"
  }, 
  "colorama-0.3.9-py27_0": {
    "md5": "15e176dd1a0d44b51d82e0c3b3df1b51", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/colorama-0.3.9-py27_0.tar.bz2"
  }, 
  "conda-4.3.21-py27_0": {
    "md5": "b847f385ddef0034cfe7ce4d6921ccda", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/conda-4.3.21-py27_0.tar.bz2"
  }, 
  "conda-env-2.6.0-0": {
    "md5": "2960d60b70d36823df7d9daea41decd0", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/conda-env-2.6.0-0.tar.bz2"
  }, 
  "configparser-3.5.0-py27_0": {
    "md5": "2ee94cf0f0fee3e7bf1c78368d79873a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/configparser-3.5.0-py27_0.tar.bz2"
  }, 
  "contextlib2-0.5.5-py27_0": {
    "md5": "65eb5d269954e7f57e6ff2851fc97afe", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/contextlib2-0.5.5-py27_0.tar.bz2"
  }, 
  "cryptography-1.8.1-py27_0": {
    "md5": "aee6f5c6766c589dcec2ff252043126e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cryptography-1.8.1-py27_0.tar.bz2"
  }, 
  "curl-7.52.1-0": {
    "md5": "c33cc04cd7aaf7a7485a58986339fec0", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/curl-7.52.1-0.tar.bz2"
  }, 
  "cycler-0.10.0-py27_0": {
    "md5": "5fa3237284832ffeaeded70c6b6953e6", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cycler-0.10.0-py27_0.tar.bz2"
  }, 
  "cython-0.25.2-py27_0": {
    "md5": "29ed9f823ddb8af5912b399bf78764fd", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cython-0.25.2-py27_0.tar.bz2"
  }, 
  "cytoolz-0.8.2-py27_0": {
    "md5": "c0f4c3199dfff0080f00a6661f6912e9", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/cytoolz-0.8.2-py27_0.tar.bz2"
  }, 
  "dask-0.14.3-py27_1": {
    "md5": "5bb2db5c0b72179906100937d70ff63e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/dask-0.14.3-py27_1.tar.bz2"
  }, 
  "datashape-0.5.4-py27_0": {
    "md5": "2ffeef6b73d534ba1156b6252f3d0ed1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/datashape-0.5.4-py27_0.tar.bz2"
  }, 
  "dbus-1.10.10-0": {
    "md5": "80493c5ee4ee933a24e8e25f9e09cdaa", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/dbus-1.10.10-0.tar.bz2"
  }, 
  "decorator-4.0.11-py27_0": {
    "md5": "39c8761c645e7522c049a73660d19739", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/decorator-4.0.11-py27_0.tar.bz2"
  }, 
  "distributed-1.16.3-py27_0": {
    "md5": "1eeeef4b7554872297f9116c40327ff5", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/distributed-1.16.3-py27_0.tar.bz2"
  }, 
  "docutils-0.13.1-py27_0": {
    "md5": "42eb2ad90e96aa5da5337c22cdcd17dc", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/docutils-0.13.1-py27_0.tar.bz2"
  }, 
  "entrypoints-0.2.2-py27_1": {
    "md5": "ce1a2914678768b1b56baf5dab799f28", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/entrypoints-0.2.2-py27_1.tar.bz2"
  }, 
  "enum34-1.1.6-py27_0": {
    "md5": "1407d00867af7f42ab47628d8639f943", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/enum34-1.1.6-py27_0.tar.bz2"
  }, 
  "et_xmlfile-1.0.1-py27_0": {
    "md5": "e14c617f590057cc25a5f23cb5af7811", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/et_xmlfile-1.0.1-py27_0.tar.bz2"
  }, 
  "expat-2.1.0-0": {
    "md5": "13df3cb2b432de77be2c13103c39692c", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/expat-2.1.0-0.tar.bz2"
  }, 
  "fastcache-1.0.2-py27_1": {
    "md5": "02f3d02f47cfb8d0a1fa828889e2b22a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/fastcache-1.0.2-py27_1.tar.bz2"
  }, 
  "flask-0.12.2-py27_0": {
    "md5": "1a84424bd822819fd2d891bcf3113ee7", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/flask-0.12.2-py27_0.tar.bz2"
  }, 
  "flask-cors-3.0.2-py27_0": {
    "md5": "640afa2457370b0a69f06896b90bf1d3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/flask-cors-3.0.2-py27_0.tar.bz2"
  }, 
  "fontconfig-2.12.1-3": {
    "md5": "39f039da0f42da4937c6e1840a86d2d0", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/fontconfig-2.12.1-3.tar.bz2"
  }, 
  "freetype-2.5.5-2": {
    "md5": "cb3923944a73c59b557cd85a8c55fcb3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/freetype-2.5.5-2.tar.bz2"
  }, 
  "funcsigs-1.0.2-py27_0": {
    "md5": "b7448f5cea660e2db1b6bf891106ac88", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/funcsigs-1.0.2-py27_0.tar.bz2"
  }, 
  "functools32-3.2.3.2-py27_0": {
    "md5": "f75142247b96c51882de66edcd2338ac", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/functools32-3.2.3.2-py27_0.tar.bz2"
  }, 
  "futures-3.1.1-py27_0": {
    "md5": "f29f72c678cebb22eebddb37463845e3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/futures-3.1.1-py27_0.tar.bz2"
  }, 
  "get_terminal_size-1.0.0-py27_0": {
    "md5": "25ad88504e366d061a845afef3c10f9d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/get_terminal_size-1.0.0-py27_0.tar.bz2"
  }, 
  "gevent-1.2.1-py27_0": {
    "md5": "c5e106b1fad2920828f0784e321f89cc", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/gevent-1.2.1-py27_0.tar.bz2"
  }, 
  "glib-2.50.2-1": {
    "md5": "a9e31763e3c9a330a3273e4f75829c15", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/glib-2.50.2-1.tar.bz2"
  }, 
  "greenlet-0.4.12-py27_0": {
    "md5": "05553217073500465139274a1907f981", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/greenlet-0.4.12-py27_0.tar.bz2"
  }, 
  "grin-1.2.1-py27_3": {
    "md5": "cccc667cfe6f813265f14782fb98008b", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/grin-1.2.1-py27_3.tar.bz2"
  }, 
  "gst-plugins-base-1.8.0-0": {
    "md5": "9c19173e0e6b8eaeadfb604000169f78", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/gst-plugins-base-1.8.0-0.tar.bz2"
  }, 
  "gstreamer-1.8.0-0": {
    "md5": "5a5e2aff3d6bd7f4d69a60140ed0a571", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/gstreamer-1.8.0-0.tar.bz2"
  }, 
  "h5py-2.7.0-np112py27_0": {
    "md5": "a955f3b0b31351b835ff198956c3cc59", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/h5py-2.7.0-np112py27_0.tar.bz2"
  }, 
  "harfbuzz-0.9.39-2": {
    "md5": "db418998fca943b8a5ce57b79497db8a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/harfbuzz-0.9.39-2.tar.bz2"
  }, 
  "hdf5-1.8.17-1": {
    "md5": "a10478d5543c24f8260d6b777c7f4bc7", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/hdf5-1.8.17-1.tar.bz2"
  }, 
  "heapdict-1.0.0-py27_1": {
    "md5": "48f42c6ae6230f702c00109990263427", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/heapdict-1.0.0-py27_1.tar.bz2"
  }, 
  "html5lib-0.999-py27_0": {
    "md5": "b767b83e135188f1c70b7bc90247082c", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/html5lib-0.999-py27_0.tar.bz2"
  }, 
  "icu-54.1-0": {
    "md5": "c1d5cbeb7127a672211dd56b28603a8f", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/icu-54.1-0.tar.bz2"
  }, 
  "idna-2.5-py27_0": {
    "md5": "680fe23c0fdf53dcbbc246ef6a5acaad", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/idna-2.5-py27_0.tar.bz2"
  }, 
  "imagesize-0.7.1-py27_0": {
    "md5": "998736603119925e8aa7eaf9d052d747", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/imagesize-0.7.1-py27_0.tar.bz2"
  }, 
  "ipaddress-1.0.18-py27_0": {
    "md5": "747ad30c034c5f3b5e8bbbd89b10bf48", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ipaddress-1.0.18-py27_0.tar.bz2"
  }, 
  "ipykernel-4.6.1-py27_0": {
    "md5": "203963aa9d64199755b2ae2cf503a000", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ipykernel-4.6.1-py27_0.tar.bz2"
  }, 
  "ipython-5.3.0-py27_0": {
    "md5": "166c3b761ec760beb9ccb21b9b9c0b6a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ipython-5.3.0-py27_0.tar.bz2"
  }, 
  "ipython_genutils-0.2.0-py27_0": {
    "md5": "02b3f9b5e92be2356b0483fed10a4611", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ipython_genutils-0.2.0-py27_0.tar.bz2"
  }, 
  "ipywidgets-6.0.0-py27_0": {
    "md5": "6648ea247cad2fffe2ee473d7f491394", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ipywidgets-6.0.0-py27_0.tar.bz2"
  }, 
  "isort-4.2.5-py27_0": {
    "md5": "7e017b59109db2ed9b6957f4b0686621", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/isort-4.2.5-py27_0.tar.bz2"
  }, 
  "itsdangerous-0.24-py27_0": {
    "md5": "48295d6fb79e3c601a611515e5d9a6f6", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/itsdangerous-0.24-py27_0.tar.bz2"
  }, 
  "jbig-2.1-0": {
    "md5": "334b102413fec962bf65c4d60697da34", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jbig-2.1-0.tar.bz2"
  }, 
  "jdcal-1.3-py27_0": {
    "md5": "1b78a01a78a9efede4108959b0b05858", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jdcal-1.3-py27_0.tar.bz2"
  }, 
  "jedi-0.10.2-py27_2": {
    "md5": "5452974cba52f7213aa052f9355169dc", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jedi-0.10.2-py27_2.tar.bz2"
  }, 
  "jinja2-2.9.6-py27_0": {
    "md5": "52b049c04079de9324ec6493a7810566", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jinja2-2.9.6-py27_0.tar.bz2"
  }, 
  "jpeg-9b-0": {
    "md5": "5e29fdb319af276a9336971b481ea03d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jpeg-9b-0.tar.bz2"
  }, 
  "jsonschema-2.6.0-py27_0": {
    "md5": "8838d8fdf663fab06806deecededa622", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jsonschema-2.6.0-py27_0.tar.bz2"
  }, 
  "jupyter-1.0.0-py27_3": {
    "md5": "5aa301ad67ff8d6bef6a94015f769c77", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jupyter-1.0.0-py27_3.tar.bz2"
  }, 
  "jupyter_client-5.0.1-py27_0": {
    "md5": "50bb2e1d87b97126d65f49599378fabc", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jupyter_client-5.0.1-py27_0.tar.bz2"
  }, 
  "jupyter_console-5.1.0-py27_0": {
    "md5": "e6a41d0d487047d6b2ebc75727a6d0c2", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jupyter_console-5.1.0-py27_0.tar.bz2"
  }, 
  "jupyter_core-4.3.0-py27_0": {
    "md5": "12cd0b00bf2eb6e776e90d1b343f692d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/jupyter_core-4.3.0-py27_0.tar.bz2"
  }, 
  "lazy-object-proxy-1.2.2-py27_0": {
    "md5": "2b15b648e51670431fefc4c030ebdfaf", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/lazy-object-proxy-1.2.2-py27_0.tar.bz2"
  }, 
  "libffi-3.2.1-1": {
    "md5": "b8297fc22fd5f713e619329001546adc", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libffi-3.2.1-1.tar.bz2"
  }, 
  "libgcc-4.8.5-2": {
    "md5": "fb7f042b7146a04e13597d7a20fe70b1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libgcc-4.8.5-2.tar.bz2"
  }, 
  "libgfortran-3.0.0-1": {
    "md5": "d7c7e92a8ccc518709474dd3eda896b9", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libgfortran-3.0.0-1.tar.bz2"
  }, 
  "libiconv-1.14-0": {
    "md5": "70cf265a73da373e088724afa51aecbf", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libiconv-1.14-0.tar.bz2"
  }, 
  "libpng-1.6.27-0": {
    "md5": "f52422bac41c66be8f28f17645fa69cc", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libpng-1.6.27-0.tar.bz2"
  }, 
  "libsodium-1.0.10-0": {
    "md5": "bdc90579f54cf2cbcd5c0ec67247cfad", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libsodium-1.0.10-0.tar.bz2"
  }, 
  "libtiff-4.0.6-3": {
    "md5": "d7e271eea139b6fb218ccbbaececa389", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libtiff-4.0.6-3.tar.bz2"
  }, 
  "libtool-2.4.2-0": {
    "md5": "69b5bc70e475344bb0c7e9d0d0382896", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libtool-2.4.2-0.tar.bz2"
  }, 
  "libxcb-1.12-1": {
    "md5": "3375e9a852d4ae539fbbb6a948eb2f7e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libxcb-1.12-1.tar.bz2"
  }, 
  "libxml2-2.9.4-0": {
    "md5": "750025fc9f5f623446bdafa7c726c61d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libxml2-2.9.4-0.tar.bz2"
  }, 
  "libxslt-1.1.29-0": {
    "md5": "221767e891fd74efe885c60b9d66923d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/libxslt-1.1.29-0.tar.bz2"
  }, 
  "llvmlite-0.18.0-py27_0": {
    "md5": "247b8bbfeb91aabca6e42485aaaedca5", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/llvmlite-0.18.0-py27_0.tar.bz2"
  }, 
  "locket-0.2.0-py27_1": {
    "md5": "b0743bf6a6bf3d830f56e7858473edc2", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/locket-0.2.0-py27_1.tar.bz2"
  }, 
  "lxml-3.7.3-py27_0": {
    "md5": "fa3a877a135e5f9222c41542e31ecdb1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/lxml-3.7.3-py27_0.tar.bz2"
  }, 
  "markupsafe-0.23-py27_2": {
    "md5": "6309f1cf8fe22fe6c09c8085e878f386", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/markupsafe-0.23-py27_2.tar.bz2"
  }, 
  "matplotlib-2.0.2-np112py27_0": {
    "md5": "3a9241323d988eb561b626f8b9b81615", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/matplotlib-2.0.2-np112py27_0.tar.bz2"
  }, 
  "mistune-0.7.4-py27_0": {
    "md5": "2634ffe80c16432bb4d96f6398a86084", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/mistune-0.7.4-py27_0.tar.bz2"
  }, 
  "mkl-2017.0.1-0": {
    "md5": "8f95cbe4dde3672bbda05e92d5e47832", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/mkl-2017.0.1-0.tar.bz2"
  }, 
  "mkl-service-1.1.2-py27_3": {
    "md5": "b8cbf6cb8da105d13ae9566d42a12115", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/mkl-service-1.1.2-py27_3.tar.bz2"
  }, 
  "mpmath-0.19-py27_1": {
    "md5": "e583fcd7efb1aaf5cac4ac31549f1f54", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/mpmath-0.19-py27_1.tar.bz2"
  }, 
  "msgpack-python-0.4.8-py27_0": {
    "md5": "2690af386487e0e690d50d5fa7291720", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/msgpack-python-0.4.8-py27_0.tar.bz2"
  }, 
  "multipledispatch-0.4.9-py27_0": {
    "md5": "2bee8b1c836430d4c364593b771edaeb", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/multipledispatch-0.4.9-py27_0.tar.bz2"
  }, 
  "navigator-updater-0.1.0-py27_0": {
    "md5": "b110f23343ad95ef15ec33da7e7c897e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/navigator-updater-0.1.0-py27_0.tar.bz2"
  }, 
  "nbconvert-5.1.1-py27_0": {
    "md5": "38b57d22b5d096cc7c37e77052c2b9a9", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/nbconvert-5.1.1-py27_0.tar.bz2"
  }, 
  "nbformat-4.3.0-py27_0": {
    "md5": "1f1c43ff5c6a4cd70c5d4c84c2dd8346", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/nbformat-4.3.0-py27_0.tar.bz2"
  }, 
  "networkx-1.11-py27_0": {
    "md5": "58015d5d4b112797b0ae0e9fa685476c", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/networkx-1.11-py27_0.tar.bz2"
  }, 
  "nltk-3.2.3-py27_0": {
    "md5": "32e9e137c48eddc75caa7f789fbdf7c8", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/nltk-3.2.3-py27_0.tar.bz2"
  }, 
  "nose-1.3.7-py27_1": {
    "md5": "2d066c04cbb6d7315b627a74f3aba8b5", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/nose-1.3.7-py27_1.tar.bz2"
  }, 
  "notebook-5.0.0-py27_0": {
    "md5": "1e67e5d89243bc7708700b56f804989e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/notebook-5.0.0-py27_0.tar.bz2"
  }, 
  "numba-0.33.0-np112py27_0": {
    "md5": "0e1cfa4c815dde79c630bf2197475f83", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/numba-0.33.0-np112py27_0.tar.bz2"
  }, 
  "numexpr-2.6.2-np112py27_0": {
    "md5": "b3c79820907c7d70be64716524919eff", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/numexpr-2.6.2-np112py27_0.tar.bz2"
  }, 
  "numpy-1.12.1-py27_0": {
    "md5": "15f10b951e15ae034810356dec55214b", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/numpy-1.12.1-py27_0.tar.bz2"
  }, 
  "numpydoc-0.6.0-py27_0": {
    "md5": "c2c0cd3b860d431b2f2fe3eb1337e29d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/numpydoc-0.6.0-py27_0.tar.bz2"
  }, 
  "odo-0.5.0-py27_1": {
    "md5": "c5571a70751613e86a658d76cec5448d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/odo-0.5.0-py27_1.tar.bz2"
  }, 
  "olefile-0.44-py27_0": {
    "md5": "759a2dcd661c5fe2ad81566f549a83a8", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/olefile-0.44-py27_0.tar.bz2"
  }, 
  "openpyxl-2.4.7-py27_0": {
    "md5": "dc6909504a92b288fef310a785e426db", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/openpyxl-2.4.7-py27_0.tar.bz2"
  }, 
  "openssl-1.0.2l-0": {
    "md5": "4b777ebd155103c6ca3c8abff32a95b6", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/openssl-1.0.2l-0.tar.bz2"
  }, 
  "packaging-16.8-py27_0": {
    "md5": "6befe0e2716ff1532fd383b4b3341898", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/packaging-16.8-py27_0.tar.bz2"
  }, 
  "pandas-0.20.1-np112py27_0": {
    "md5": "91791fd95bccf6baeeb47dbab86acae3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pandas-0.20.1-np112py27_0.tar.bz2"
  }, 
  "pandocfilters-1.4.1-py27_0": {
    "md5": "c6eacc7cfd4d18f76a3a18bb9edda256", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pandocfilters-1.4.1-py27_0.tar.bz2"
  }, 
  "pango-1.40.3-1": {
    "md5": "f7a3e32b5d5e79cf600ccff5844f2631", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pango-1.40.3-1.tar.bz2"
  }, 
  "partd-0.3.8-py27_0": {
    "md5": "3dab2aca87a860cc612768e904dd120e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/partd-0.3.8-py27_0.tar.bz2"
  }, 
  "path.py-10.3.1-py27_0": {
    "md5": "8fc0f60622d6310ea7bdb9bddc6459a9", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/path.py-10.3.1-py27_0.tar.bz2"
  }, 
  "pathlib2-2.2.1-py27_0": {
    "md5": "5cce337e05bd7ba0a7c1d117c8c6472c", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pathlib2-2.2.1-py27_0.tar.bz2"
  }, 
  "patsy-0.4.1-py27_0": {
    "md5": "b78c3447b7cfab154b81571d2b011879", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/patsy-0.4.1-py27_0.tar.bz2"
  }, 
  "pcre-8.39-1": {
    "md5": "62c472936cea126dfb939ae5a1388004", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pcre-8.39-1.tar.bz2"
  }, 
  "pep8-1.7.0-py27_0": {
    "md5": "f05d5198f0cc6830b624a732094ed965", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pep8-1.7.0-py27_0.tar.bz2"
  }, 
  "pexpect-4.2.1-py27_0": {
    "md5": "ca6e4c4af19348e28804af0faa55f401", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pexpect-4.2.1-py27_0.tar.bz2"
  }, 
  "pickleshare-0.7.4-py27_0": {
    "md5": "f60435aca432d6f1ec0c88df78b7ffc1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pickleshare-0.7.4-py27_0.tar.bz2"
  }, 
  "pillow-4.1.1-py27_0": {
    "md5": "124aa2c2f4bdc716badeb467eed75397", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pillow-4.1.1-py27_0.tar.bz2"
  }, 
  "pip-9.0.1-py27_1": {
    "md5": "f5266886d7cf83558b806762f2278145", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pip-9.0.1-py27_1.tar.bz2"
  }, 
  "pixman-0.34.0-0": {
    "md5": "f97f025b386f0f482714261e4d1f66dd", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pixman-0.34.0-0.tar.bz2"
  }, 
  "ply-3.10-py27_0": {
    "md5": "40833b7b0e405286730ab01f22a5059b", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ply-3.10-py27_0.tar.bz2"
  }, 
  "prompt_toolkit-1.0.14-py27_0": {
    "md5": "a62881fc9cc26955cd18088cd06ebd64", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/prompt_toolkit-1.0.14-py27_0.tar.bz2"
  }, 
  "psutil-5.2.2-py27_0": {
    "md5": "2b0ead8df38bdb3234d8dbae74864d4e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/psutil-5.2.2-py27_0.tar.bz2"
  }, 
  "ptyprocess-0.5.1-py27_0": {
    "md5": "4c1006e594173fb0eccb8fcd7b9ab1ad", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ptyprocess-0.5.1-py27_0.tar.bz2"
  }, 
  "py-1.4.33-py27_0": {
    "md5": "8faa815636cffdd52cb4b679971fa2c0", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/py-1.4.33-py27_0.tar.bz2"
  }, 
  "pycairo-1.10.0-py27_0": {
    "md5": "fad9b73e4e310fb1a2169d78452ce9ec", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pycairo-1.10.0-py27_0.tar.bz2"
  }, 
  "pycosat-0.6.2-py27_0": {
    "md5": "b89cd441838ecaeba16f9fa92767f8c3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pycosat-0.6.2-py27_0.tar.bz2"
  }, 
  "pycparser-2.17-py27_0": {
    "md5": "fdf8bb112ff1c21e370651f080dcc737", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pycparser-2.17-py27_0.tar.bz2"
  }, 
  "pycrypto-2.6.1-py27_6": {
    "md5": "9120f6cb17ff3ea189e2055a6ebcb50f", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pycrypto-2.6.1-py27_6.tar.bz2"
  }, 
  "pycurl-7.43.0-py27_2": {
    "md5": "04a52e5225f5da361f94ca0ff377e443", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pycurl-7.43.0-py27_2.tar.bz2"
  }, 
  "pyflakes-1.5.0-py27_0": {
    "md5": "057cb45885f087b8ad1d324d5bbae046", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pyflakes-1.5.0-py27_0.tar.bz2"
  }, 
  "pygments-2.2.0-py27_0": {
    "md5": "7dc68f9276beee6d6c0728ae3f51fd18", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pygments-2.2.0-py27_0.tar.bz2"
  }, 
  "pylint-1.6.4-py27_1": {
    "md5": "6bae82f0f1214f886fadd777da31a02d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pylint-1.6.4-py27_1.tar.bz2"
  }, 
  "pyodbc-4.0.16-py27_0": {
    "md5": "dd2e21eba132260e925643348300b80a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pyodbc-4.0.16-py27_0.tar.bz2"
  }, 
  "pyopenssl-17.0.0-py27_0": {
    "md5": "f1da8b388617ca9f390ee869911fdfb2", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pyopenssl-17.0.0-py27_0.tar.bz2"
  }, 
  "pyparsing-2.1.4-py27_0": {
    "md5": "8aaec959dfb65042e4dff555ffb83c26", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pyparsing-2.1.4-py27_0.tar.bz2"
  }, 
  "pyqt-5.6.0-py27_2": {
    "md5": "80cc234372e1512884875ccb01d1cb21", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pyqt-5.6.0-py27_2.tar.bz2"
  }, 
  "pytables-3.3.0-np112py27_0": {
    "md5": "97afc2fcf57a9b6fb7db09fa21053c49", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pytables-3.3.0-np112py27_0.tar.bz2"
  }, 
  "pytest-3.0.7-py27_0": {
    "md5": "1c0908e07506b94469d42d4c76408df1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pytest-3.0.7-py27_0.tar.bz2"
  }, 
  "python-2.7.13-0": {
    "md5": "75517b0f5176e9cf704befa1cac3be23", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/python-2.7.13-0.tar.bz2"
  }, 
  "python-dateutil-2.6.0-py27_0": {
    "md5": "7dac7f35c1dd37dae460bde5cc341800", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/python-dateutil-2.6.0-py27_0.tar.bz2"
  }, 
  "pytz-2017.2-py27_0": {
    "md5": "4640efa8adde9bc89c9ca51d27c95d3d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pytz-2017.2-py27_0.tar.bz2"
  }, 
  "pywavelets-0.5.2-np112py27_0": {
    "md5": "8e5bed47d5b0d47dd4e90d4fd56c27d6", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pywavelets-0.5.2-np112py27_0.tar.bz2"
  }, 
  "pyyaml-3.12-py27_0": {
    "md5": "02c068a0959abe5a83b15a2b022d886a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pyyaml-3.12-py27_0.tar.bz2"
  }, 
  "pyzmq-16.0.2-py27_0": {
    "md5": "d0a2f253cbf2370a2371d3b2a7757391", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/pyzmq-16.0.2-py27_0.tar.bz2"
  }, 
  "qt-5.6.2-4": {
    "md5": "a4ac470407bfc7cb2e3cc1882b28c253", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/qt-5.6.2-4.tar.bz2"
  }, 
  "qtawesome-0.4.4-py27_0": {
    "md5": "b2298e6f7654d5bd620a923b25adc9bd", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/qtawesome-0.4.4-py27_0.tar.bz2"
  }, 
  "qtconsole-4.3.0-py27_0": {
    "md5": "c10e4f4a1cff270e412b217e91bbe42a", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/qtconsole-4.3.0-py27_0.tar.bz2"
  }, 
  "qtpy-1.2.1-py27_0": {
    "md5": "a2a5a1206f889e48df5b3ea68bf85945", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/qtpy-1.2.1-py27_0.tar.bz2"
  }, 
  "readline-6.2-2": {
    "md5": "d050607fb2934282470d06872e0e6cce", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/readline-6.2-2.tar.bz2"
  }, 
  "requests-2.14.2-py27_0": {
    "md5": "e73b74d2a6179a4e9822bd22108417ec", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/requests-2.14.2-py27_0.tar.bz2"
  }, 
  "rope-0.9.4-py27_1": {
    "md5": "e6017d8b755b05462880c7d30cf581e1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/rope-0.9.4-py27_1.tar.bz2"
  }, 
  "ruamel_yaml-0.11.14-py27_1": {
    "md5": "b5af08fe28821a11348785c10ec9cc23", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ruamel_yaml-0.11.14-py27_1.tar.bz2"
  }, 
  "scandir-1.5-py27_0": {
    "md5": "fa7080c32b5a353e94fed11a56c1e334", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/scandir-1.5-py27_0.tar.bz2"
  }, 
  "scikit-image-0.13.0-np112py27_0": {
    "md5": "e92e89f144f6adad1ee69a7edbb43dbf", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/scikit-image-0.13.0-np112py27_0.tar.bz2"
  }, 
  "scikit-learn-0.18.1-np112py27_1": {
    "md5": "c99c45ba4a614dd6f4eb1eebb5cb0c20", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/scikit-learn-0.18.1-np112py27_1.tar.bz2"
  }, 
  "scipy-0.19.0-np112py27_0": {
    "md5": "7c04a78df69a24cdde303e07f19d95f3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/scipy-0.19.0-np112py27_0.tar.bz2"
  }, 
  "seaborn-0.7.1-py27_0": {
    "md5": "977652a76370bc94e42293dc53e78e59", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/seaborn-0.7.1-py27_0.tar.bz2"
  }, 
  "setuptools-27.2.0-py27_0": {
    "md5": "96b639928547b4f837f7d4d1ed5f2d7e", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/setuptools-27.2.0-py27_0.tar.bz2"
  }, 
  "simplegeneric-0.8.1-py27_1": {
    "md5": "419ef3d6235392a86d24867981306107", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/simplegeneric-0.8.1-py27_1.tar.bz2"
  }, 
  "singledispatch-3.4.0.3-py27_0": {
    "md5": "95ce90450ee1287efc3f3f7c70efc990", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/singledispatch-3.4.0.3-py27_0.tar.bz2"
  }, 
  "sip-4.18-py27_0": {
    "md5": "d8443d26dcaf83587f21eedb9c447da2", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/sip-4.18-py27_0.tar.bz2"
  }, 
  "six-1.10.0-py27_0": {
    "md5": "78d229dce568dd4bac1d88328f822fc5", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/six-1.10.0-py27_0.tar.bz2"
  }, 
  "snowballstemmer-1.2.1-py27_0": {
    "md5": "0834f6476d1d750a3654346a7dad8747", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/snowballstemmer-1.2.1-py27_0.tar.bz2"
  }, 
  "sortedcollections-0.5.3-py27_0": {
    "md5": "62877ee43d3366eb6c312d2a1f657b1d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/sortedcollections-0.5.3-py27_0.tar.bz2"
  }, 
  "sortedcontainers-1.5.7-py27_0": {
    "md5": "5d23961d685399b14fd82740b5d14fd7", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/sortedcontainers-1.5.7-py27_0.tar.bz2"
  }, 
  "sphinx-1.5.6-py27_0": {
    "md5": "49c2f9c6fef95b1ab9e42d31549dd274", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/sphinx-1.5.6-py27_0.tar.bz2"
  }, 
  "spyder-3.1.4-py27_0": {
    "md5": "250fd42543b932e3f20dec3a75aafc96", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/spyder-3.1.4-py27_0.tar.bz2"
  }, 
  "sqlalchemy-1.1.9-py27_0": {
    "md5": "a3a9e306fa4c6bc6b9f6b8bf00333524", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/sqlalchemy-1.1.9-py27_0.tar.bz2"
  }, 
  "sqlite-3.13.0-0": {
    "md5": "ceb2d63b0dc2ec8a2e1da9378f07ab98", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/sqlite-3.13.0-0.tar.bz2"
  }, 
  "ssl_match_hostname-3.4.0.2-py27_1": {
    "md5": "ffca0f9278ac34828b183820aa868c5d", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/ssl_match_hostname-3.4.0.2-py27_1.tar.bz2"
  }, 
  "statsmodels-0.8.0-np112py27_0": {
    "md5": "a85f4624bb4dea5231542cb1139bfc30", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/statsmodels-0.8.0-np112py27_0.tar.bz2"
  }, 
  "subprocess32-3.2.7-py27_0": {
    "md5": "cb23915417fb0c4d519ba82790b1c932", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/subprocess32-3.2.7-py27_0.tar.bz2"
  }, 
  "sympy-1.0-py27_0": {
    "md5": "e96b1b72f0e023184d6c32f620932ae9", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/sympy-1.0-py27_0.tar.bz2"
  }, 
  "tblib-1.3.2-py27_0": {
    "md5": "2bc12b17388947f96ad739a964a66708", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/tblib-1.3.2-py27_0.tar.bz2"
  }, 
  "terminado-0.6-py27_0": {
    "md5": "9f4870fb0d8ece71313189019b1c4163", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/terminado-0.6-py27_0.tar.bz2"
  }, 
  "testpath-0.3-py27_0": {
    "md5": "10c60a50acdca85f7b66a715c5dbe983", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/testpath-0.3-py27_0.tar.bz2"
  }, 
  "tk-8.5.18-0": {
    "md5": "902f0fd689a01a835c9e69aefbe58fdd", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/tk-8.5.18-0.tar.bz2"
  }, 
  "toolz-0.8.2-py27_0": {
    "md5": "688a6b00fa4fa50a20353f63fb53dbc9", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/toolz-0.8.2-py27_0.tar.bz2"
  }, 
  "tornado-4.5.1-py27_0": {
    "md5": "6ec42ac09490e6f49655a2ee96fb77ef", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/tornado-4.5.1-py27_0.tar.bz2"
  }, 
  "traitlets-4.3.2-py27_0": {
    "md5": "b82c25e69d0ce547e16867d4537fa5df", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/traitlets-4.3.2-py27_0.tar.bz2"
  }, 
  "unicodecsv-0.14.1-py27_0": {
    "md5": "17f5af29c443f26fa707b901aa54b8e6", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/unicodecsv-0.14.1-py27_0.tar.bz2"
  }, 
  "unixodbc-2.3.4-0": {
    "md5": "9b2fadd3dd9c9fac66f52b3bc184a119", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/unixodbc-2.3.4-0.tar.bz2"
  }, 
  "wcwidth-0.1.7-py27_0": {
    "md5": "2b9c7073f1210540923cf36f6053eb61", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/wcwidth-0.1.7-py27_0.tar.bz2"
  }, 
  "werkzeug-0.12.2-py27_0": {
    "md5": "4063711b6e86fb48130e43dfc72c11cc", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/werkzeug-0.12.2-py27_0.tar.bz2"
  }, 
  "wheel-0.29.0-py27_0": {
    "md5": "35cf0da93616ea8c495ce1d143316d7c", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/wheel-0.29.0-py27_0.tar.bz2"
  }, 
  "widgetsnbextension-2.0.0-py27_0": {
    "md5": "cc46906f836cbac19945af84cbf844c3", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/widgetsnbextension-2.0.0-py27_0.tar.bz2"
  }, 
  "wrapt-1.10.10-py27_0": {
    "md5": "3030ee9938f4acd6a4eb06235e4846cf", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/wrapt-1.10.10-py27_0.tar.bz2"
  }, 
  "xlrd-1.0.0-py27_0": {
    "md5": "e6a6cacdeb0f68b76913f8699c5ca6c1", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/xlrd-1.0.0-py27_0.tar.bz2"
  }, 
  "xlsxwriter-0.9.6-py27_0": {
    "md5": "f6b29854b2e089ff7c337057bc4ef5ae", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/xlsxwriter-0.9.6-py27_0.tar.bz2"
  }, 
  "xlwt-1.2.0-py27_0": {
    "md5": "0cc240a111b95ff9c52693c5283a750c", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/xlwt-1.2.0-py27_0.tar.bz2"
  }, 
  "xz-5.2.2-1": {
    "md5": "6352a951a56d7e96a49cbcb3abdda7ed", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/xz-5.2.2-1.tar.bz2"
  }, 
  "yaml-0.1.6-0": {
    "md5": "dac36570434ddc9e16e54709c114bd96", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/yaml-0.1.6-0.tar.bz2"
  }, 
  "zeromq-4.1.5-0": {
    "md5": "07fe48d7c52cfb01eb38d13ac7ccc237", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/zeromq-4.1.5-0.tar.bz2"
  }, 
  "zict-0.1.2-py27_0": {
    "md5": "fda26e19bb53a0b041d32ae6f0051040", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/zict-0.1.2-py27_0.tar.bz2"
  }, 
  "zlib-1.2.8-3": {
    "md5": "60d5ea874984e4c750f187a26c827382", 
    "url": "https://repo.continuum.io/pkgs/free/linux-64/zlib-1.2.8-3.tar.bz2"
  }
}
C_ENVS = {
  "root": [
    "python-2.7.13-0", 
    "_license-1.1-py27_1", 
    "alabaster-0.7.10-py27_0", 
    "anaconda-client-1.6.3-py27_0", 
    "anaconda-navigator-1.6.2-py27_0", 
    "anaconda-project-0.6.0-py27_0", 
    "asn1crypto-0.22.0-py27_0", 
    "astroid-1.4.9-py27_0", 
    "astropy-1.3.2-np112py27_0", 
    "babel-2.4.0-py27_0", 
    "backports-1.0-py27_0", 
    "backports_abc-0.5-py27_0", 
    "beautifulsoup4-4.6.0-py27_0", 
    "bitarray-0.8.1-py27_0", 
    "blaze-0.10.1-py27_0", 
    "bleach-1.5.0-py27_0", 
    "bokeh-0.12.5-py27_1", 
    "boto-2.46.1-py27_0", 
    "bottleneck-1.2.1-np112py27_0", 
    "cairo-1.14.8-0", 
    "cdecimal-2.3-py27_2", 
    "cffi-1.10.0-py27_0", 
    "chardet-3.0.3-py27_0", 
    "click-6.7-py27_0", 
    "cloudpickle-0.2.2-py27_0", 
    "clyent-1.2.2-py27_0", 
    "colorama-0.3.9-py27_0", 
    "configparser-3.5.0-py27_0", 
    "contextlib2-0.5.5-py27_0", 
    "cryptography-1.8.1-py27_0", 
    "curl-7.52.1-0", 
    "cycler-0.10.0-py27_0", 
    "cython-0.25.2-py27_0", 
    "cytoolz-0.8.2-py27_0", 
    "dask-0.14.3-py27_1", 
    "datashape-0.5.4-py27_0", 
    "dbus-1.10.10-0", 
    "decorator-4.0.11-py27_0", 
    "distributed-1.16.3-py27_0", 
    "docutils-0.13.1-py27_0", 
    "entrypoints-0.2.2-py27_1", 
    "enum34-1.1.6-py27_0", 
    "et_xmlfile-1.0.1-py27_0", 
    "expat-2.1.0-0", 
    "fastcache-1.0.2-py27_1", 
    "flask-0.12.2-py27_0", 
    "flask-cors-3.0.2-py27_0", 
    "fontconfig-2.12.1-3", 
    "freetype-2.5.5-2", 
    "funcsigs-1.0.2-py27_0", 
    "functools32-3.2.3.2-py27_0", 
    "futures-3.1.1-py27_0", 
    "get_terminal_size-1.0.0-py27_0", 
    "gevent-1.2.1-py27_0", 
    "glib-2.50.2-1", 
    "greenlet-0.4.12-py27_0", 
    "grin-1.2.1-py27_3", 
    "gst-plugins-base-1.8.0-0", 
    "gstreamer-1.8.0-0", 
    "h5py-2.7.0-np112py27_0", 
    "harfbuzz-0.9.39-2", 
    "hdf5-1.8.17-1", 
    "heapdict-1.0.0-py27_1", 
    "html5lib-0.999-py27_0", 
    "icu-54.1-0", 
    "idna-2.5-py27_0", 
    "imagesize-0.7.1-py27_0", 
    "ipaddress-1.0.18-py27_0", 
    "ipykernel-4.6.1-py27_0", 
    "ipython-5.3.0-py27_0", 
    "ipython_genutils-0.2.0-py27_0", 
    "ipywidgets-6.0.0-py27_0", 
    "isort-4.2.5-py27_0", 
    "itsdangerous-0.24-py27_0", 
    "jbig-2.1-0", 
    "jdcal-1.3-py27_0", 
    "jedi-0.10.2-py27_2", 
    "jinja2-2.9.6-py27_0", 
    "jpeg-9b-0", 
    "jsonschema-2.6.0-py27_0", 
    "jupyter-1.0.0-py27_3", 
    "jupyter_client-5.0.1-py27_0", 
    "jupyter_console-5.1.0-py27_0", 
    "jupyter_core-4.3.0-py27_0", 
    "lazy-object-proxy-1.2.2-py27_0", 
    "libffi-3.2.1-1", 
    "libgcc-4.8.5-2", 
    "libgfortran-3.0.0-1", 
    "libiconv-1.14-0", 
    "libpng-1.6.27-0", 
    "libsodium-1.0.10-0", 
    "libtiff-4.0.6-3", 
    "libtool-2.4.2-0", 
    "libxcb-1.12-1", 
    "libxml2-2.9.4-0", 
    "libxslt-1.1.29-0", 
    "llvmlite-0.18.0-py27_0", 
    "locket-0.2.0-py27_1", 
    "lxml-3.7.3-py27_0", 
    "markupsafe-0.23-py27_2", 
    "matplotlib-2.0.2-np112py27_0", 
    "mistune-0.7.4-py27_0", 
    "mkl-2017.0.1-0", 
    "mkl-service-1.1.2-py27_3", 
    "mpmath-0.19-py27_1", 
    "msgpack-python-0.4.8-py27_0", 
    "multipledispatch-0.4.9-py27_0", 
    "navigator-updater-0.1.0-py27_0", 
    "nbconvert-5.1.1-py27_0", 
    "nbformat-4.3.0-py27_0", 
    "networkx-1.11-py27_0", 
    "nltk-3.2.3-py27_0", 
    "nose-1.3.7-py27_1", 
    "notebook-5.0.0-py27_0", 
    "numba-0.33.0-np112py27_0", 
    "numexpr-2.6.2-np112py27_0", 
    "numpy-1.12.1-py27_0", 
    "numpydoc-0.6.0-py27_0", 
    "odo-0.5.0-py27_1", 
    "olefile-0.44-py27_0", 
    "openpyxl-2.4.7-py27_0", 
    "openssl-1.0.2l-0", 
    "packaging-16.8-py27_0", 
    "pandas-0.20.1-np112py27_0", 
    "pandocfilters-1.4.1-py27_0", 
    "pango-1.40.3-1", 
    "partd-0.3.8-py27_0", 
    "path.py-10.3.1-py27_0", 
    "pathlib2-2.2.1-py27_0", 
    "patsy-0.4.1-py27_0", 
    "pcre-8.39-1", 
    "pep8-1.7.0-py27_0", 
    "pexpect-4.2.1-py27_0", 
    "pickleshare-0.7.4-py27_0", 
    "pillow-4.1.1-py27_0", 
    "pip-9.0.1-py27_1", 
    "pixman-0.34.0-0", 
    "ply-3.10-py27_0", 
    "prompt_toolkit-1.0.14-py27_0", 
    "psutil-5.2.2-py27_0", 
    "ptyprocess-0.5.1-py27_0", 
    "py-1.4.33-py27_0", 
    "pycairo-1.10.0-py27_0", 
    "pycosat-0.6.2-py27_0", 
    "pycparser-2.17-py27_0", 
    "pycrypto-2.6.1-py27_6", 
    "pycurl-7.43.0-py27_2", 
    "pyflakes-1.5.0-py27_0", 
    "pygments-2.2.0-py27_0", 
    "pylint-1.6.4-py27_1", 
    "pyodbc-4.0.16-py27_0", 
    "pyopenssl-17.0.0-py27_0", 
    "pyparsing-2.1.4-py27_0", 
    "pyqt-5.6.0-py27_2", 
    "pytables-3.3.0-np112py27_0", 
    "pytest-3.0.7-py27_0", 
    "python-dateutil-2.6.0-py27_0", 
    "pytz-2017.2-py27_0", 
    "pywavelets-0.5.2-np112py27_0", 
    "pyyaml-3.12-py27_0", 
    "pyzmq-16.0.2-py27_0", 
    "qt-5.6.2-4", 
    "qtawesome-0.4.4-py27_0", 
    "qtconsole-4.3.0-py27_0", 
    "qtpy-1.2.1-py27_0", 
    "readline-6.2-2", 
    "requests-2.14.2-py27_0", 
    "rope-0.9.4-py27_1", 
    "ruamel_yaml-0.11.14-py27_1", 
    "scandir-1.5-py27_0", 
    "scikit-image-0.13.0-np112py27_0", 
    "scikit-learn-0.18.1-np112py27_1", 
    "scipy-0.19.0-np112py27_0", 
    "seaborn-0.7.1-py27_0", 
    "setuptools-27.2.0-py27_0", 
    "simplegeneric-0.8.1-py27_1", 
    "singledispatch-3.4.0.3-py27_0", 
    "sip-4.18-py27_0", 
    "six-1.10.0-py27_0", 
    "snowballstemmer-1.2.1-py27_0", 
    "sortedcollections-0.5.3-py27_0", 
    "sortedcontainers-1.5.7-py27_0", 
    "sphinx-1.5.6-py27_0", 
    "spyder-3.1.4-py27_0", 
    "sqlalchemy-1.1.9-py27_0", 
    "sqlite-3.13.0-0", 
    "ssl_match_hostname-3.4.0.2-py27_1", 
    "statsmodels-0.8.0-np112py27_0", 
    "subprocess32-3.2.7-py27_0", 
    "sympy-1.0-py27_0", 
    "tblib-1.3.2-py27_0", 
    "terminado-0.6-py27_0", 
    "testpath-0.3-py27_0", 
    "tk-8.5.18-0", 
    "toolz-0.8.2-py27_0", 
    "tornado-4.5.1-py27_0", 
    "traitlets-4.3.2-py27_0", 
    "unicodecsv-0.14.1-py27_0", 
    "unixodbc-2.3.4-0", 
    "wcwidth-0.1.7-py27_0", 
    "werkzeug-0.12.2-py27_0", 
    "wheel-0.29.0-py27_0", 
    "widgetsnbextension-2.0.0-py27_0", 
    "wrapt-1.10.10-py27_0", 
    "xlrd-1.0.0-py27_0", 
    "xlsxwriter-0.9.6-py27_0", 
    "xlwt-1.2.0-py27_0", 
    "xz-5.2.2-1", 
    "yaml-0.1.6-0", 
    "zeromq-4.1.5-0", 
    "zict-0.1.2-py27_0", 
    "zlib-1.2.8-3", 
    "anaconda-4.4.0-np112py27_0", 
    "conda-4.3.21-py27_0", 
    "conda-env-2.6.0-0"
  ]
}



def _link(src, dst, linktype=LINK_HARD):
    if on_win:
        raise NotImplementedError

    if linktype == LINK_HARD:
        os.link(src, dst)
    elif linktype == LINK_COPY:
        # copy relative symlinks as symlinks
        if islink(src) and not os.readlink(src).startswith('/'):
            os.symlink(os.readlink(src), dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise Exception("Did not expect linktype=%r" % linktype)


def rm_rf(path):
    """
    try to delete path, but never fail
    """
    try:
        if islink(path) or isfile(path):
            # Note that we have to check if the destination is a link because
            # exists('/path/to/dead-link') will return False, although
            # islink('/path/to/dead-link') is True.
            os.unlink(path)
        elif isdir(path):
            shutil.rmtree(path)
    except (OSError, IOError):
        pass


def yield_lines(path):
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        yield line


prefix_placeholder = ('/opt/anaconda1anaconda2'
                      # this is intentionally split into parts,
                      # such that running this program on itself
                      # will leave it unchanged
                      'anaconda3')

def read_has_prefix(path):
    """
    reads `has_prefix` file and return dict mapping filenames to
    tuples(placeholder, mode)
    """
    import shlex

    res = {}
    try:
        for line in yield_lines(path):
            try:
                placeholder, mode, f = [x.strip('"\'') for x in
                                        shlex.split(line, posix=False)]
                res[f] = (placeholder, mode)
            except ValueError:
                res[line] = (prefix_placeholder, 'text')
    except IOError:
        pass
    return res


def exp_backoff_fn(fn, *args):
    """
    for retrying file operations that fail on Windows due to virus scanners
    """
    if not on_win:
        return fn(*args)

    import time
    import errno
    max_tries = 6  # max total time = 6.4 sec
    for n in range(max_tries):
        try:
            result = fn(*args)
        except (OSError, IOError) as e:
            if e.errno in (errno.EPERM, errno.EACCES):
                if n == max_tries - 1:
                    raise Exception("max_tries=%d reached" % max_tries)
                time.sleep(0.1 * (2 ** n))
            else:
                raise e
        else:
            return result


class PaddingError(Exception):
    pass


def binary_replace(data, a, b):
    """
    Perform a binary replacement of `data`, where the placeholder `a` is
    replaced with `b` and the remaining string is padded with null characters.
    All input arguments are expected to be bytes objects.
    """
    def replace(match):
        occurances = match.group().count(a)
        padding = (len(a) - len(b)) * occurances
        if padding < 0:
            raise PaddingError(a, b, padding)
        return match.group().replace(a, b) + b'\0' * padding

    pat = re.compile(re.escape(a) + b'([^\0]*?)\0')
    res = pat.sub(replace, data)
    assert len(res) == len(data)
    return res


def update_prefix(path, new_prefix, placeholder, mode):
    if on_win:
        # force all prefix replacements to forward slashes to simplify need
        # to escape backslashes - replace with unix-style path separators
        new_prefix = new_prefix.replace('\\', '/')

    path = os.path.realpath(path)
    with open(path, 'rb') as fi:
        data = fi.read()
    if mode == 'text':
        new_data = data.replace(placeholder.encode('utf-8'),
                                new_prefix.encode('utf-8'))
    elif mode == 'binary':
        if on_win:
            # anaconda-verify will not allow binary placeholder on Windows.
            # However, since some packages might be created wrong (and a
            # binary placeholder would break the package, we just skip here.
            return
        new_data = binary_replace(data, placeholder.encode('utf-8'),
                                  new_prefix.encode('utf-8'))
    else:
        sys.exit("Invalid mode:" % mode)

    if new_data == data:
        return
    st = os.lstat(path)
    # unlink in case the file is memory mapped
    exp_backoff_fn(os.unlink, path)
    with open(path, 'wb') as fo:
        fo.write(new_data)
    os.chmod(path, stat.S_IMODE(st.st_mode))


def name_dist(dist):
    return dist.rsplit('-', 2)[0]


def create_meta(prefix, dist, info_dir, extra_info):
    """
    Create the conda metadata, in a given prefix, for a given package.
    """
    # read info/index.json first
    with open(join(info_dir, 'index.json')) as fi:
        meta = json.load(fi)
    # add extra info
    meta.update(extra_info)
    # write into <prefix>/conda-meta/<dist>.json
    meta_dir = join(prefix, 'conda-meta')
    if not isdir(meta_dir):
        os.makedirs(meta_dir)
        with open(join(meta_dir, 'history'), 'w') as fo:
            fo.write('')
    with open(join(meta_dir, dist + '.json'), 'w') as fo:
        json.dump(meta, fo, indent=2, sort_keys=True)


def run_script(prefix, dist, action='post-link'):
    """
    call the post-link (or pre-unlink) script, and return True on success,
    False on failure
    """
    path = join(prefix, 'Scripts' if on_win else 'bin', '.%s-%s.%s' % (
            name_dist(dist),
            action,
            'bat' if on_win else 'sh'))
    if not isfile(path):
        return True
    if SKIP_SCRIPTS:
        print("WARNING: skipping %s script by user request" % action)
        return True

    if on_win:
        try:
            args = [os.environ['COMSPEC'], '/c', path]
        except KeyError:
            return False
    else:
        shell_path = '/bin/sh' if 'bsd' in sys.platform else '/bin/bash'
        args = [shell_path, path]

    env = os.environ
    env['PREFIX'] = prefix

    import subprocess
    try:
        subprocess.check_call(args, env=env)
    except subprocess.CalledProcessError:
        return False
    return True


url_pat = re.compile(r'''
(?P<baseurl>\S+/)                 # base URL
(?P<fn>[^\s#/]+)                  # filename
([#](?P<md5>[0-9a-f]{32}))?       # optional MD5
$                                 # EOL
''', re.VERBOSE)

def read_urls(dist):
    try:
        data = open(join(PKGS_DIR, 'urls')).read()
        for line in data.split()[::-1]:
            m = url_pat.match(line)
            if m is None:
                continue
            if m.group('fn') == '%s.tar.bz2' % dist:
                return {'url': m.group('baseurl') + m.group('fn'),
                        'md5': m.group('md5')}
    except IOError:
        pass
    return {}


def read_no_link(info_dir):
    res = set()
    for fn in 'no_link', 'no_softlink':
        try:
            res.update(set(yield_lines(join(info_dir, fn))))
        except IOError:
            pass
    return res


def linked(prefix):
    """
    Return the (set of canonical names) of linked packages in prefix.
    """
    meta_dir = join(prefix, 'conda-meta')
    if not isdir(meta_dir):
        return set()
    return set(fn[:-5] for fn in os.listdir(meta_dir) if fn.endswith('.json'))


def link(prefix, dist, linktype=LINK_HARD):
    '''
    Link a package in a specified prefix.  We assume that the packacge has
    been extra_info in either
      - <PKGS_DIR>/dist
      - <ROOT_PREFIX>/ (when the linktype is None)
    '''
    if linktype:
        source_dir = join(PKGS_DIR, dist)
        info_dir = join(source_dir, 'info')
        no_link = read_no_link(info_dir)
    else:
        info_dir = join(prefix, 'info')

    files = list(yield_lines(join(info_dir, 'files')))
    has_prefix_files = read_has_prefix(join(info_dir, 'has_prefix'))

    if linktype:
        for f in files:
            src = join(source_dir, f)
            dst = join(prefix, f)
            dst_dir = dirname(dst)
            if not isdir(dst_dir):
                os.makedirs(dst_dir)
            if exists(dst):
                if FORCE:
                    rm_rf(dst)
                else:
                    raise Exception("dst exists: %r" % dst)
            lt = linktype
            if f in has_prefix_files or f in no_link or islink(src):
                lt = LINK_COPY
            try:
                _link(src, dst, lt)
            except OSError:
                pass

    for f in sorted(has_prefix_files):
        placeholder, mode = has_prefix_files[f]
        try:
            update_prefix(join(prefix, f), prefix, placeholder, mode)
        except PaddingError:
            sys.exit("ERROR: placeholder '%s' too short in: %s\n" %
                     (placeholder, dist))

    if not run_script(prefix, dist, 'post-link'):
        sys.exit("Error: post-link failed for: %s" % dist)

    meta = {
        'files': files,
        'link': ({'source': source_dir,
                  'type': link_name_map.get(linktype)}
                 if linktype else None),
    }
    try:    # add URL and MD5
        meta.update(IDISTS[dist])
    except KeyError:
        meta.update(read_urls(dist))
    meta['installed_by'] = 'Anaconda2-4.4.0-Linux-x86_64'
    create_meta(prefix, dist, info_dir, meta)


def duplicates_to_remove(linked_dists, keep_dists):
    """
    Returns the (sorted) list of distributions to be removed, such that
    only one distribution (for each name) remains.  `keep_dists` is an
    interable of distributions (which are not allowed to be removed).
    """
    from collections import defaultdict

    keep_dists = set(keep_dists)
    ldists = defaultdict(set) # map names to set of distributions
    for dist in linked_dists:
        name = name_dist(dist)
        ldists[name].add(dist)

    res = set()
    for dists in ldists.values():
        # `dists` is the group of packages with the same name
        if len(dists) == 1:
            # if there is only one package, nothing has to be removed
            continue
        if dists & keep_dists:
            # if the group has packages which are have to be kept, we just
            # take the set of packages which are in group but not in the
            # ones which have to be kept
            res.update(dists - keep_dists)
        else:
            # otherwise, we take lowest (n-1) (sorted) packages
            res.update(sorted(dists)[:-1])
    return sorted(res)


def remove_duplicates():
    idists = []
    for line in open(join(PKGS_DIR, 'urls')):
        m = url_pat.match(line)
        if m:
            fn = m.group('fn')
            idists.append(fn[:-8])

    keep_files = set()
    for dist in idists:
        with open(join(ROOT_PREFIX, 'conda-meta', dist + '.json')) as fi:
            meta = json.load(fi)
        keep_files.update(meta['files'])

    for dist in duplicates_to_remove(linked(ROOT_PREFIX), idists):
        print("unlinking: %s" % dist)
        meta_path = join(ROOT_PREFIX, 'conda-meta', dist + '.json')
        with open(meta_path) as fi:
            meta = json.load(fi)
        for f in meta['files']:
            if f not in keep_files:
                rm_rf(join(ROOT_PREFIX, f))
        rm_rf(meta_path)


def link_idists():
    src = join(PKGS_DIR, 'urls')
    dst = join(ROOT_PREFIX, '.hard-link')
    assert isfile(src), src
    assert not isfile(dst), dst
    try:
        _link(src, dst, LINK_HARD)
        linktype = LINK_HARD
    except OSError:
        linktype = LINK_COPY
    finally:
        rm_rf(dst)

    for env_name in sorted(C_ENVS):
        dists = C_ENVS[env_name]
        assert isinstance(dists, list)
        if len(dists) == 0:
            continue

        prefix = prefix_env(env_name)
        for dist in dists:
            assert dist in IDISTS
            link(prefix, dist, linktype)

        for dist in duplicates_to_remove(linked(prefix), dists):
            meta_path = join(prefix, 'conda-meta', dist + '.json')
            print("WARNING: unlinking: %s" % meta_path)
            try:
                os.rename(meta_path, meta_path + '.bak')
            except OSError:
                rm_rf(meta_path)


def prefix_env(env_name):
    if env_name == 'root':
        return ROOT_PREFIX
    else:
        return join(ROOT_PREFIX, 'envs', env_name)


def post_extract(env_name='root'):
    """
    assuming that the package is extracted in the environment `env_name`,
    this function does everything link() does except the actual linking,
    i.e. update prefix files, run 'post-link', creates the conda metadata,
    and removed the info/ directory afterwards.
    """
    prefix = prefix_env(env_name)
    info_dir = join(prefix, 'info')
    with open(join(info_dir, 'index.json')) as fi:
        meta = json.load(fi)
    dist = '%(name)s-%(version)s-%(build)s' % meta
    if FORCE:
        run_script(prefix, dist, 'pre-unlink')
    link(prefix, dist, linktype=None)
    shutil.rmtree(info_dir)


def main():
    global ROOT_PREFIX, PKGS_DIR

    p = OptionParser(description="conda link tool used by installers")

    p.add_option('--root-prefix',
                 action="store",
                 default=abspath(join(__file__, '..', '..')),
                 help="root prefix (defaults to %default)")

    p.add_option('--post',
                 action="store",
                 help="perform post extract (on a single package), "
                      "in environment NAME",
                 metavar='NAME')

    opts, args = p.parse_args()
    if args:
        p.error('no arguments expected')

    ROOT_PREFIX = opts.root_prefix.replace('//', '/')
    PKGS_DIR = join(ROOT_PREFIX, 'pkgs')

    if opts.post:
        post_extract(opts.post)
        return

    if FORCE:
        print("using -f (force) option")

    link_idists()


def main2():
    global SKIP_SCRIPTS

    p = OptionParser(description="conda post extract tool used by installers")

    p.add_option('--skip-scripts',
                 action="store_true",
                 help="skip running pre/post-link scripts")

    p.add_option('--rm-dup',
                 action="store_true",
                 help="remove duplicates")

    opts, args = p.parse_args()
    if args:
        p.error('no arguments expected')

    if opts.skip_scripts:
        SKIP_SCRIPTS = True

    if opts.rm_dup:
        remove_duplicates()
        return

    post_extract()


def warn_on_special_chrs():
    if on_win:
        return
    for c in SPECIAL_ASCII:
        if c in ROOT_PREFIX:
            print("WARNING: found '%s' in install prefix." % c)


if __name__ == '__main__':
    if IDISTS:
        main()
        warn_on_special_chrs()
    else: # common usecase
        main2()
