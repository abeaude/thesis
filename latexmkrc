# Settings
$ENV{'TEXINPUTS'}='./texmf//:';
$pdf_mode = 5;
$postscript_mode = $dvi_mode = 0;
$allow_subdir_creation=2;
$max_repeat=6;
$bibtex_use = 2;
$xdvipdfmx_silent_switch = ""; # -q
$xdvipdfmx = "xdvipdfmx -E -o %D %O %S"; #-z 0
$success_cmd = "texlogfilter --no-box --no-info --no-filename %Y/%A.log | sed -r 's/\\x1B\\\[(;?[0-9]{1,3})+[mGK]//g' | grep -vE 'LaTeX2e|Document Class|Output written|Package silence' && if [ %A = 'main' ]; then rm -f %Y/%A.log %Y/%A.aux; fi";
$failure_cmd = "texlogfilter --no-box --no-info --no-filename %Y/%A.log | sed -r 's/\\x1B\\\[(;?[0-9]{1,3})+[mGK]//g' | grep -vE 'LaTeX2e|Document Class|Output written|Package silence' ";
$silent = 1;
# $makeindex = "makeindex %O -o %D %S";
# $out_dir, $aux_dir, $out2_dir, @out2_exts, $xdvipdfmx
set_tex_cmds( '--shell-escape %O %S' );
push @generated_exts, 'loe', 'lol', 'lor', 'run.xml', 'glg', 'glstex', 'glo', 'bcf', 'fls', 'glg-abr', 'glo-abr', 'ist', 'lof', 'slg', 'slo', 'sls', 'toc', 'fdb_latexmk', 'gls', 'gls-abr', 'xdv', 'aux';

##############
# Glossaries #
##############
add_cus_dep( 'glo', 'gls', 0, 'glo2gls' );
add_cus_dep( 'acn', 'acr', 0, 'glo2gls');  # from Overleaf v1
sub glo2gls {
    my ($base_name, $path) = fileparse( $_[0] );
    my @args = ( "-d", $path, $base_name );
    if ($silent) { unshift @args, "-q"; }
    return system "makeglossaries", @args;
}

#############
# makeindex #
#############
@ist = glob("*.ist");
if (scalar(@ist) > 0) {
    $makeindex = "makeindex -s $ist[0] %O -o %D %S";
}

#######
# svg #
#######
# add_cus_dep('svg', 'pdf', 0, 'svg2pdf');
sub svg2pdf {
    system("inkscape --export-area-drawing --export-pdf=\"$_[0].pdf\" \"$_[0].svg\"");
}
##########
# drawio #
##########
add_cus_dep('drawio', 'pdf', 0, 'drawio2pdf');
sub drawio2pdf {
    system("drawio --export --format pdf --border 0 --crop --page-index 1 \"$_[0].drawio\"");
}

add_cus_dep( 'tex', 'pdf', 0, 'makerobustexternalize' );
sub makerobustexternalize {
    if ( $root_filename ne $_[0] )  {
        print "Compiling external document ", $_[0], "\n";
        my ($base_name, $path) = fileparse( $_[0] );
        system "cd $path && xelatex -interaction=batchmode -halt-on-error $base_name.tex";
    } else {
        print "Not running on main file", "\n";
    }
}
add_cus_dep( 'tex', 'aux', 0, 'makeexternaldocument' );
sub makeexternaldocument {
    system "pwd";
    if ( $root_filename ne $_[0] && $root_filename ne "main" )  {
        print "Compiling external document ", $_[0], "\n";
        my ($base_name, $path) = fileparse( $_[0] );
        system "xelatex --shell-escape -no-pdf -synctex=1 -interaction=batchmode -file-line-error -recorder -output-directory='chapters/PDF' $_[0].tex && biber --quiet $path/PDF/$base_name.bcf && xelatex --shell-escape -no-pdf -synctex=1 -interaction=batchmode -file-line-error -recorder -output-directory='chapters/PDF' $_[0].tex && xelatex --shell-escape -no-pdf -synctex=1 -interaction=batchmode -file-line-error -recorder -output-directory='chapters/PDF' $_[0].tex";
        rdb_add_generated( "$path/PDF/$base_name.aux" );
        copy "$path/PDF/$base_name.aux", "chapters/";
        popd();
    } 
}

if( ($ENV{GITHUB_ACTIONS} // "false") eq "true" ){
    print "Not running `git_info_2` in github actions.";
} else {
    do './perl/gitinfo2.pm';
}
print "";
####################
# externaldocument #
####################
# # Subdirectory for output files from compilation of external documents:
# $sub_doc_output = '../PDF/output-subdoc';

# # Options to supply to latexmk for compilation of external documents:
# @sub_doc_options = ();

# push @sub_doc_options, '-pdfxe'; # Use xelatex for compilation of external documents.

# #--------------------

# # Add a pattern for xr's log-file message about missing files to latexmk's
# # list.  Latexmk's variable @file_not_found is not yet documented.
# # This line isn't necessary for v. 4.80 or later of latexmk.
# push @file_not_found, '^No file\\s*(.+)\s*$';

# add_cus_dep( 'tex', 'aux', 0, 'makeexternaldocument' );
# sub makeexternaldocument {
#     if ( $root_filename ne $_[0] )  {
#         my ($base_name, $path) = fileparse( $_[0] );
#         print $base_name;
#         print $path;
#         pushd $path;
#         my $return = system "latexmk",
#                             @sub_doc_options,
#                             "-aux-directory=$sub_doc_output",
#                             "-output-directory=$sub_doc_output",
#                             $base_name;
#         if ( ($sub_doc_output ne ' ') && ($sub_doc_output ne '.') ) {
#                # In this case, .aux file generated by pdflatex isn't in same
#                # directory as the .tex file.
#                # Therefore:
#                # 1. Actual generated aux file must be listed as produced by this
#                #    rule, so that latexmk deals with dependencies correctly.
#                #    (Problem to overcome: If $sub_dir_output is same as $aux_dir
#                #    for the main document, xr may read the .aux file in the
#                #    aux_dir rather than the one the cus dep is assumed by latexmk
#                #    to produce, which is in the same directory as the .tex source
#                #    file for this custom dependency.)
#                #    Use the latexmk subroutine rdb_add_generated to do this.
#                # 2. A copy of the .aux file must be in same directory as .tex file
#                #    to satisfy latexmk's definition of a custom dependency.
#              rdb_add_generated( "$sub_doc_output/$base_name.aux" );
#              copy "$sub_doc_output/$base_name.aux", ".";
#         }
#         popd();
#         return $return;
#    }
# }
