version=$1
# 1.)
echo "1.) create documentation directory"
echo "1.) sphinx-apidoc RootInteractive -o ./docs"
rm -rf docs/*
rm -fr index.rst  conf.py _build*
cd ../
sphinx-apidoc RootInteractive -o docs -d 3 -P -A "Marian Ivanov" -V $version
cd docs
sphinx-quickstart -p RootInteractive  -a "Marian Ivanov" -v 0.0.09 -l "en"  --ext-autodoc --ext-doctest  --ext-intersphinx  --ext-todo  --ext-coverage --ext-imgmath --ext-mathjax  --ext-ifconfig  --ext-viewcode --ext-githubpages
# --ext-autodoc         enable autodoc extension
#  --ext-doctest         enable doctest extension
#  --ext-intersphinx     enable intersphinx extension
#  --ext-todo            enable todo extension
#  --ext-coverage        enable coverage extension
#  --ext-imgmath         enable imgmath extension
#  --ext-mathjax         enable mathjax extension
#  --ext-ifconfig        enable ifconfig extension
#  --ext-viewcode        enable viewcode extension
#  --ext-githubpages     enable githubpages extension
# 2.)
#echo "2.) cp sphinx/conf.py docs/"
#cp ../sphinx/conf.py .
htmlSetup=$(cat <<-END
# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'
html_theme_options = {
    "rightsidebar": "false",
    "relbarbgcolor": "black",
    "collapsiblesidebar": "true",
    "externalrefs":"true",
    "stickysidebar":"false",
    "body_max_width": "none"
}
END
)
echo  "$htmlSetup" >> conf.py
# 3.)
echo "3.) make html"
make  html
cd ../
# 4.)
echo " 4.) synchronize to server"
echo "rsync -rvzt  docs/_build/* /eos/user/r/rootinteractive/www/"
rsync -rvzt  docs/_build/* /eos/user/r/rootinteractive/www/

