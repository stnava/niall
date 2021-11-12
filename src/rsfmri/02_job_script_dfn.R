system("sudo apt-get -y install pandoc")
outfn=tempfile( fileext='.html' )
print(outfn)
rmarkdown::render("~/coderepo/niall/src/rsfmri/03_dfn.Rmd",output_file=outfn)
