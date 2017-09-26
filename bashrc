kpython(){
  docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it kaggle/python python "$@"
}
ikpython() {
  docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it kaggle/python ipython
}
kjupyter() {
  (sleep 3 && open "http://$(docker-machine ip docker2):8888")&
  docker run -v $PWD:/tmp/working -w=/tmp/working -p 8888:8888 --rm -it kaggle/python jupyter notebook --no-browser --ip="\*" --notebook-dir=/tmp/working
}
