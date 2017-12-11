Example command for running `create_plots.py`:
```
python create_plots.py -p test -e .png
```

This will create the files `test_p-at-k.png` and `test_ndcg-at-k.png` in the current directory.

* To view the plots in the output window, set the flag `-v` or `--verbose`.
* To change DPI of plots, use `--dpi` option. The default is set to 1200, which is really high but could make the program run slightly slower. 
* Run script with `-h` or `--help` for help
