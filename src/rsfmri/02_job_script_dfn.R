system("sudo apt-get install pandoc")
outfn=tempfile( fileext='.html' )
print(outfn)
rmarkdown::render("~/code/niall/src/rsfmri/03_dfn.Rmd",output_file=outfn)
