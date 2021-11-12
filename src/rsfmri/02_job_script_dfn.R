system("sudo apt-get -y install pandoc")
outfn=paste0( "~/html/", basename( tempfile( fileext='.html' ) ) )
print(outfn)
rmarkdown::render("~/coderepo/niall/src/rsfmri/03_dfn.Rmd",output_file=outfn)
