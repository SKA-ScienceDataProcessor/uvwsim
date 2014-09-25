
pythonlibs=@PROJECT_BINARY_DIR@/python
if [ ${PYTHONPATH/$pythonlibs} = $PYTHONPATH ]; then
    if [ -z "$PYTHONPATH" ]; then
        export PYTHONPATH=$pythonlibs
    else
        export PYTHONPATH=$pythonlibs:$PYTHONPATH
    fi
fi
python @PROJECT_BINARY_DIR@/test/test_python_interface.py