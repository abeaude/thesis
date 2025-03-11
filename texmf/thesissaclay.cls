\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesissaclay}[2014/08/16 Example LaTeX class]
\overfullrule=2cm
\newif\if@gitmark\@gitmarkfalse
%%%%%%%%%%%%%
%% OPTIONS %%
%%%%%%%%%%%%%
%\DeclareOption{onecolumn}{\OptionNotUsed}
\DeclareOption{web}{
	\PassOptionsToClass{oneside}{scrbook}
	\PassOptionsToPackage{autooneside=off}{scrlayer-scrpage}
	\PassOptionsToPackage{linkcolor=black}{hyperref}
}
\DeclareOption{print}{
	\PassOptionsToClass{twoside,open=right}{scrbook}
	\PassOptionsToPackage{linkcolor=black}{hyperref}%pdfpagelayout=TwoPageRight,
}
\DeclareOption{final}{
	\PassOptionsToPackage{final}{showlabels}
	\PassOptionsToPackage{off}{mindflow}
	\overfullrule=0cm
}
\DeclareOption{english}{\global\def\@lang{english}}
\DeclareOption{gitmark}{\@gitmarktrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ProcessOptions\relax

%%%%%%%%%%%%%%%%%
%% SETUP CLASS %%
%%%%%%%%%%%%%%%%%
\RequirePackage[debrief, save]{silence}
\WarningFilter{latexfont}{Some font shapes were not available}
\WarningFilter{latexfont}{Font shape}
\WarningFilter{hyperref}{Token not allowed in a PDF string}
\WarningFilter{caption}{The option `hypcap=true' will be ignored for this particular}
\WarningFilter{latex}{Writing or overwriting file}
\WarningFilter{fvextra}{csquotes should be loaded after}
\LoadClass[fontsize=12pt, a4paper, titlepage=firstiscover, cleardoublepage=empty, appendixprefix=true, parskip=half, DIV=11]{scrbook}
\KOMAoption{toc}{listof, index, bibliography, graduated}
\RequirePackage[nodisplayskipstretch]{setspace}
\setstretch{1.2}
\AfterTOCHead{\singlespacing}
\KOMAoptions{DIV=last}
\RequirePackage{scrhack}
\RequirePackage[pagewise]{lineno}
\RequirePackage{ragged2e}
\RequirePackage{xspace}
%%%%%%%%%%%%%%
%% PACKAGES %%
%%%%%%%%%%%%%%
%% FONTS %%
\RequirePackage[no-math]{fontspec}
\setmainfont{OpenSans}[
	Extension = .ttf,
	Path = fonts/,
	StylisticSet={1,2,3,4},
	Ligatures=Common,
	UprightFont = *-Regular,
	BoldFont = *-Bold,
	ItalicFont = *-Italic,
	BoldItalicFont = *-BoldItalic
]
\newfontfamily\opensans{OpenSans}[
	Extension = .ttf,
	Path = fonts/,
	UprightFont = *-Regular,
	BoldFont = *-Bold,
	ItalicFont = *-Italic,
	BoldItalicFont = *-BoldItalic
]
%\addfontfeature{LetterSpace = 15em plus.1em minus.1em}
\RequirePackage{xunicode}
\RequirePackage{microtype}
\RequirePackage[strict, autostyle=true]{csquotes}
%% LANG %%
\RequirePackage{polyglossia}
\ifdefstring{\@lang}{english}{
	\setmainlanguage{english}
	\setotherlanguage{french}
}{
	\setmainlanguage{french}
	\setotherlanguage{english}
}
\RequirePackage{textcomp}
\RequirePackage{newunicodechar}
\newfontface{\dejavusans}{DejaVuSans}

%% TABLES %%
\RequirePackage{tabularray}
\UseTblrLibrary{booktabs}
\UseTblrLibrary{diagbox}
\UseTblrLibrary{siunitx}
\RequirePackage{rotating}
\RequirePackage{makecell}
\renewcommand\theadfont{}
\RequirePackage{booktabs}
\RequirePackage{multirow}
% \RequirePackage{colortbl}
% \RequirePackage{tabularx}
% \RequirePackage{longtable}
%% MATHS %%
\RequirePackage{siunitx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
\RequirePackage{nicefrac}
% \RequirePackage[all,warning]{onlyamsmath}
\RequirePackage[math-style=TeX, bold-style=TeX, active-frac=small]{unicode-math}
\setmathfont{latinmodern-math.otf}[Scale=MatchLowercase, Path = fonts/, version=normal]
\setmathfontface\mathrm{latinmodern-math.otf}[Scale=MatchLowercase, Path = fonts/, version=normal]
\setmathfont{latinmodern-math.otf}[Scale=1, Path = fonts/, version=smaller]
\setmathfontface\mathrm{latinmodern-math.otf}[Scale=1, Path = fonts/, version=smaller]
\setmathfont{latinmodern-math.otf}[Scale=0.6, Path = fonts/, version=tiny]
\setmathfontface\mathrm{latinmodern-math.otf}[Scale=0.6, Path = fonts/, version=tiny]
\setmathfont{latinmodern-math.otf}[Scale=0.8, Path = fonts/, version=middlesize]
\setmathfontface\mathrm{latinmodern-math.otf}[Scale=0.8, Path = fonts/, version=middlesize]
\setoperatorfont{\mathrm}
% \setoperatorfont{\symbfsfup}
\newtheorem{definition}{Definition} %TODO: language support
\newtheorem{property}{Property}
\newtheorem{theorem}{Theorem}
\newtheorem{proof}{Proof}
%% COLORS %%
\RequirePackage{fontawesome5}
\RequirePackage[table]{xcolor}
%% INTERNAL %%
\RequirePackage{etoolbox}
\RequirePackage{keyval}
\RequirePackage{nameref} % Access name of a reference
\RequirePackage{etoc} % Multiple and cusomize table of contents
\RequirePackage[inline]{showlabels} % Show label keys when writing the document, do not work with subcaption
\RequirePackage{functional}
\RequirePackage{morewrites}
\AtBeginEnvironment{table}{\small}
\AtBeginEnvironment{longtable}{\scriptsize}
%% GRAPHICS %%
\RequirePackage{graphicx} % Required for inserting images
\RequirePackage{wrapfig2}
\RequirePackage{tikz}
\RequirePackage{robust-externalize}
\RequirePackage{pgfplots}
\RequirePackage{pgfplotstable}
\RequirePackage{subcaption}
\renewcommand{\thesubfigure}{\Alph{subfigure}}
\renewcommand{\thesubtable}{\Alph{subtable}}
\DeclareCaptionLabelFormat{bf-parens}{(\textbf{#2})}
\DeclareCaptionLabelFormat{bf-sub}{\textbf{#2}}
\subcaptionsetup{subrefformat=bf-parens, labelformat=bf-sub}
\newcommand{\subrefrange}[2]{\subcaption@reffmt\p@subref{\ref{sub@#1}--\ref{sub@#2}}}
%% OTHER %%
\RequirePackage{lipsum}
\RequirePackage[framemethod=tex]{mdframed}
\RequirePackage[object=vectorian]{pgfornament}
\RequirePackage{tcolorbox}
\PassOptionsToPackage{outputdir=PDF}{minted}
\tcbuselibrary{skins,minted}
\RequirePackage[]{mindflow}
\RequirePackage[all]{nowidow}
\RequirePackage[hyphenation, parindent]{impnattypo}
\RequirePackage[inline]{enumitem}
%\RequirePackage[newfloat=true]{minted}
\renewcommand{\MintedPygmentize}{/root/venv/bin/pygmentize}
\usemintedstyle{flexoki-style}
\RequirePackage[algoruled, algochapter]{algorithm2e}
\RequirePackage[l3]{csvsimple}
\RequirePackage{lscape}
\RequirePackage{xstring}
\RequirePackage{eqparbox}
\RequirePackage{relsize}
\RequirePackage[local, dirty={with uncommited changes!}]{gitinfo2}
\RequirePackage{datetime2}
\RequirePackage{pifont}
\newcommand{\cmark}{\ding{51}\xspace}%
\newcommand{\xmark}{\ding{55}\xspace}%
%\RequirePackage[french, backgroundcolor=B5, textcolor=white, linecolor=B5, bordercolor=A5]{todonotes}
\RequirePackage{multicol}
\RequirePackage[shortcuts]{extdash}
%% BIBLIOGRAPHY %%
\RequirePackage[backend=biber, bibstyle=alphabetic, citestyle=alphabetic, sorting=nty,backref=true, refsection=chapter, arxiv=abs, url=false, isbn=false, date=year, eprint=false, maxbibnames=5, maxcitenames=1,minbibnames=3]{biblatex}%block=space/par/nbpar/ragged
\renewcommand{\bibfont}{\small}
%% LAYOUT %%
\RequirePackage[pass]{geometry}
\RequirePackage[headsepline]{scrlayer-scrpage}
%% HYPERLINKS %%
\RequirePackage[hyperindex=true, bookmarks=true, breaklinks=true,bookmarksnumbered=true]{hyperref}
\RequirePackage[ocgcolorlinks]{ocgx2}
%% GLOSSARIES %%
\RequirePackage[symbols, abbreviations, indexcrossrefs=true]{glossaries-extra}%automake
\renewcommand{\glossarypreamble}{\footnotesize}
%% MULTIPLE DOCS %%
\RequirePackage{subfiles}
%\RequirePackage{glossaries-french}
\setabbreviationstyle[acronym]{long-short}
\makeglossaries
% conditional set based on main language
\renewcommand{\printsymbols}{\printunsrtglossary[type=symbols,title=Notations]}
%\renewcommand{\abbreviationsname}{Liste des abréviations et sigles}
%% HYPERREF SETUP %%
\AtEndPreamble{%
	\hypersetup{linktoc=all, colorlinks=true, citecolor=A2, filecolor=black, menucolor=black, runcolor=black, urlcolor=B3, pdftitle={[en]\@title, [fr]\@subtitle}, pdfauthor=\@author, pdfsubject=\@subject, pdfcreator=XeLaTeX, pdfproducer={xetex (TeX Live 2024)},pdfkeywords=\@keywordsfr, pdfdisplaydoctitle=true}%linkcolor = internal links
} %allcolors=black, hidelinks,
%% MUST BE LOADED LAST %%
\AtEndPreamble{%
	\RequirePackage[capitalise, nameinlink]{cleveref}
	\crefname{figure}{Figure}{Figures}
	\Crefname{figure}{Figure}{Figures}
	\crefname{appendixfigure}{Appendix figure}{Appendix figures}
	\crefname{appendixtable}{Appendix table}{Appendix tables}
	\crefrangelabelformat{subfigure}{\StrCompare{#1}{#2}[\lastcommonpos]\StrLeft{#1}{\number\numexpr\lastcommonpos-1}[\MyFigNumber]#3\MyFigNumber\,\crefstripprefix{#2}{#1}--\crefstripprefix{#1}{#2}#4}
	\newtcbinputlisting[
		auto counter, number within=chapter, crefname={Listing}{Listings}
	]{\pythoncode}[3][]{
		listing file={#3},
		listing only,
		top=1mm, bottom=1mm,
		right=2mm, left=2mm,
		fonttitle=\small,
		title=\textbf{Listing~\thetcbcounter\ ---}\ #2,
		minted language=python,
		minted options={breaklines, autogobble, numberblanklines=false, fontsize=\footnotesize, fontfamily=SourceCode},
		colback=codebg,
		colframe=codeframe,
		#1
	}
	\newtcolorbox[auto counter, number within=chapter, crefname={see Box}{see Boxes}]{mybox}[2][]{title=Box~\thetcbcounter: #2, fontupper=\footnotesize, fontlower=\footnotesize, colframe=Prune, #1}
	\BeforeBeginEnvironment{mybox}{
		\begin{singlespace}
			\mathversion{smaller}
			% \setmathfont{latinmodern-math.otf}[Scale=1, Path = fonts/]
			% \setmathfontface\mathrm{latinmodern-math.otf}[Scale=1, Path = fonts/]
			%\setoperatorfont{\mathrm}
			}
			\AfterEndEnvironment{mybox}{
			\mathversion{normal}
			% \setmathfont{latinmodern-math.otf}[Scale=MatchLowercase, Path = fonts/]
			% \setmathfontface\mathrm{latinmodern-math.otf}[Scale=MatchLowercase, Path = fonts/]
			%\setoperatorfont{\mathrm}
		\end{singlespace}
	}
}

% customize caption
\KOMAoption{captions}{tableheading, figuresignature, nooneline}
\setcaphanging
%\setcapmargin[1cm]{0pt}
\setcaptionalignment{J}
%\setcaptionalignment[table]{C}
\setkomafont{captionlabel}{\bfseries\footnotesize}
\setkomafont{caption}{\footnotesize}
\setkomafont{pageheadfoot}{\small\normalcolor}
% chapterentry, chapterentrydots, chapterentrypagenumber
% pagehead, pagefoot
% partentry, sectionentry
% 
% see floatrowbytocbasic and floatrow
%\newfontfamily{\lmm}{latinmodern-math.otf}[Scale=1, Path = fonts/]
\newfontfamily{\sourcecode}{MonaspaceNeon}[
	Extension=.otf,
	Scale=1,
	Path=fonts/,
	StylisticSet={1,2,3,4,5,6,7,8,9},
	Ligatures=Common,
	UprightFont = *-Regular,
	BoldFont = *-Bold,
	FontFace={b}{n}{*-Bold},
	NFSSFamily=SourceCode]
%%%%%%%%%%%%
%% COLORS %%
%%%%%%%%%%%%
\definecolor{Prune}{RGB}{99,0,60}
\definecolor{B1}{RGB}{49,62,72}
\definecolor{C1}{RGB}{124,135,143}
\definecolor{D1}{RGB}{213,218,223}
\definecolor{A2}{RGB}{198,11,70}
\definecolor{B2}{RGB}{237,20,91}
\definecolor{C2}{RGB}{238,52,35}
\definecolor{D2}{RGB}{243,115,32}
\definecolor{A3}{RGB}{124,42,144}
\definecolor{B3}{RGB}{125,106,175}
\definecolor{C3}{RGB}{198,103,29}
\definecolor{D3}{RGB}{254,188,24}
\definecolor{A4}{RGB}{0,78,125}
\definecolor{B4}{RGB}{14,135,201}
\definecolor{C4}{RGB}{0,148,181}
\definecolor{D4}{RGB}{70,195,210}
\definecolor{A5}{RGB}{0,128,122}
\definecolor{B5}{RGB}{64,183,105}
\definecolor{C5}{RGB}{140,198,62}
\definecolor{D5}{RGB}{213,223,61}

\renewcommand{\showlabelfont}{\scriptsize\slshape\color{Prune}}
%%%%%%%%%%%%%%%
%% LANGUAGES %%
%%%%%%%%%%%%%%%
\addto\captionsfrench{%
	\renewcommand{\figurename}{Figure}%
	\renewcommand{\tablename}{Tableau}%
}
\renewcommand*{\captionformat}{\ ---\ }
\renewcommand*{\figureformat}{\figurename~\thefigure}
\renewcommand*{\tableformat}{\tablename~\thetable}
%%%%%%%%%%%%%%%%%%%%%
%% CUSTOM COMMANDS %%
%%%%%%%%%%%%%%%%%%%%%
\ExplSyntaxOn
\prg_generate_conditional_variant:Nnn \tl_if_empty:n { e } { TF }
\let \IfTokenListEmpty = \tl_if_empty:eTF
\ExplSyntaxOff
% longtblr personalization
\DefTblrTemplate{caption-tag}{default}{%
	\usekomafont{captionlabel}{\tablename\hspace{0.25em}\thetable}
}
\DefTblrTemplate{caption-sep}{default}{%
	\usekomafont{captionlabel}{\captionformat}\enskip
}
\DefTblrTemplate{caption-text}{default}{%
	\usekomafont{caption}{\InsertTblrText{caption}}
}
\DefTblrTemplate{conthead-text}{default}{%
	\usekomafont{caption}{(Continued)}
}

\DefTblrTemplate{firsthead}{default}{%
	\addtocounter{table}{-1}%
	\captionsetup{type=table}%
	\IfTokenListEmpty{\InsertTblrText{entry}}{%
		\caption{\InsertTblrText{caption}}%
	}{%    
		\caption[\InsertTblrText{entry}]{\InsertTblrText{caption}}%
	}%
}

\DefTblrTemplate{capcont}{default}{%
	\UseTblrTemplate{caption-tag}{default}%
	\UseTblrTemplate{conthead-text}{default}%
}
\DefTblrTemplate{middlehead,lasthead}{default}{
	\UseTblrTemplate{capcont}{default}
}
%\DefTblrTemplate{note-sep}{default}{}

\DefTblrTemplate{note}{default}{
	\begin{multicols}{2}
		\begin{singlespace}
			\MapTblrNotes{
				\noindent
				\UseTblrTemplate{note-tag}{default}
				\UseTblrTemplate{note-text}{default}
				\par
			}
		\end{singlespace}
	\end{multicols}

}
\SetTblrStyle{contfoot-text}{font=\small}
\SetTblrStyle{note-text}{font=\footnotesize}
\SetTblrTemplate{caption-lot}{empty}
%\ProvideTotalTColorBox{}{}{}{}
\ProvideTotalTColorBox{\chapterpubli}{ v }{%
	title=Publication, %
	colframe=Prune, %
	fonttitle=\bfseries,% 
	before skip=0cm, %
	after skip=0cm,%
	left=2mm, right=2mm,
	middle=1mm, top=1mm, bottom=1mm,
}{%
	\footnotesize\faFile*[regular]~\fullcite{#1}\tcblower\scriptsize\citefield{#1}{abstract}
}
\setcounter{tocdepth}{\subsectiontocdepth}

	\defbibheading{subbibliography}{%
\section*{\Cref{refsection:\therefsection}: \nameref{refsection:\therefsection}}%
	}
	\preto{\section}{\def\leveltitle{\sectiontitle}}
	\pretocmd{\@sect}
	{\expandafter\gdef\leveltitle{#8}}
	{}{}

	\newcounter{myunitcount}
	\setcounter{myunitcount}{0}

	%% JURY MACROS %%
	\def\@jury{\begingroup\footnotesize\begin{tabular}{|p{9cm}l}\arrayrulecolor{Prune}}
			\define@key{jurymemeber}{firstname}{\def\@juryfirstname{#1}}
			\define@key{jurymemeber}{lastname}{\def\@jurylastname{#1}}
			\define@key{jurymemeber}{title}{\def\@jurytitle{#1}}
			\define@key{jurymemeber}{affiliation}{\def\@juryaffiliation{#1}}
			\define@key{jurymemeber}{role}{\def\@juryrole{#1}}
			\newcommand{\jurymember}[1]{
				\setkeys{jurymemeber}{#1}
			\protected@xappto\@jury{\textbf{\@juryfirstname} \textbf{\@jurylastname} & \@juryrole \\ \@jurytitle, \@juryaffiliation & \\}
			}
			\AtEndPreamble{\gappto\@jury{\end{tabular}\endgroup}}
	%% KEYWORDS %%
	\def\@keywordsfr{}
	\def\@keywordsen{}
	\def\keywordsfr#1{\global\def\@keywordsfr{#1}}
	\def\keywordsen#1{\global\def\@keywordsen{#1}}
	%% LOCATION%%
	\def\@location{}
	\def\location#1{\global\def\@location{#1}}
	%% ED GS REFEREE %%
	\def\@doctoralschool{}
	\def\@ednum{}
	\def\@edname{}
	\def\@logoed{}
	\def\doctoralschool#1{
		\global\def\@doctoralschool{#1}
		\ifstrequal{#1}{STIC}{
			\global\def\@ednum{580}
			\global\def\@edname{Sciences et technologies de l’information et de la communication}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{AAIF}{
			\global\def\@ednum{127}
			\global\def\@edname{Astronomie et astrophysique d'Île-de-France}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{SEIF}{
			\global\def\@ednum{129}
			\global\def\@edname{Sciences de l'environnement d’Île-de-France}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{PIF}{
			\global\def\@ednum{564}
			\global\def\@edname{Physique en Île-de-France}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{SSMMH}{
			\global\def\@ednum{566}
			\global\def\@edname{Sciences du sport, de la motricité et du mouvement humain}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{SEVE}{
			\global\def\@ednum{567}
			\global\def\@edname{Sciences du végétal : du gène à l'écosystème}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{Biosigne}{
			\global\def\@ednum{568}
			\global\def\@edname{Signalisations et réseaux intégratifs en biologie}
			\global\def\@logoed{logos/logo\_usp\_\MakeUppercase{#1}.png}
		}{}
		\ifstrequal{#1}{ITFA}{
			\global\def\@ednum{569}
			\global\def\@edname{Innovation thérapeutique : du fondamental à l'appliqué}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{EDSP}{
			\global\def\@ednum{570}
			\global\def\@edname{Santé publique}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{2MIB}{
			\global\def\@ednum{571}
			\global\def\@edname{Sciences chimiques : molécules, matériaux, instrumentation et biosystèmes}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{EDOM}{
			\global\def\@ednum{572}
			\global\def\@edname{Ondes et matière}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{INTERFACES}{
			\global\def\@ednum{573}
			\global\def\@edname{Interfaces : matériaux, systèmes, usages}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{EDMH}{
			\global\def\@ednum{574}
			\global\def\@edname{Mathématiques Hadamard}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{EOBE}{
			\global\def\@ednum{575}
			\global\def\@edname{Electrical, optical, bio : physics and engineering}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{PHENICS}{
			\global\def\@ednum{576}
			\global\def\@edname{Particules hadrons énergie et noyau : instrumentation, imagerie, cosmos et simulation}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{SDSV}{
			\global\def\@ednum{577}
			\global\def\@edname{Structure et dynamique des systèmes vivants}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{SMEMaG}{
			\global\def\@ednum{579}
			\global\def\@edname{Sciences mécaniques et énergétiques, matériaux et géosciences}
			\global\def\@logoed{logos/logo\_usp\_\MakeUppercase{#1}.png}
		}{}
		\ifstrequal{#1}{ABIES}{
			\global\def\@ednum{581}
			\global\def\@edname{Agriculture, alimentation, biologie, environnement, santé}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{CBMS}{
			\global\def\@ednum{582}
			\global\def\@edname{Cancérologie : biologie - médecine - santé}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{SSH}{
			\global\def\@ednum{629}
			\global\def\@edname{Sciences sociales et humanités}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
		\ifstrequal{#1}{DEM}{
			\global\def\@ednum{630}
			\global\def\@edname{Droit, Économie, Management}
			\global\def\@logoed{logos/logo\_usp\_#1.png}
		}{}
	}
	\def\@referee{}
	\def\referee#1{\global\def\@referee{#1}}
	\def\@graduateschool{}
	\def\graduateschool#1{\global\def\@graduateschool{#1}}
	\def\@nnt{}
	\def\nnt#1{\global\def\@nnt{#1}}

	%% TEXT THESIS PROCEEDINGS %%
	\def\@textsupervisor{}
	\def\unit#1{
		\ifnumcomp{\value{myunitcount}}{=}{0}{\gappto\@textsupervisor{\textbf{#1}}}{\gappto\@textsupervisor{ et\\\textbf{#1}}}
		\stepcounter{myunitcount}
	}
	\def\thesisdirector#1{
		\gappto\@textsupervisor{, sous la direction de\\#1}
	}
	\def\thesiscodirector#1{
		\gappto\@textsupervisor{, la co-direction de\\#1}
	}
	\def\thesissupervisorun#1{
		\gappto\@textsupervisor{, le co-encadrement de\\#1}
	}
	\def\thesissupervisordeux#1{
		\gappto\@textsupervisor{, et de\\#1}
	}
	\def\thesisindustrysupervisor#1{
		\gappto\@textsupervisor{, la co-supervision de\\#1}
	}
	\AtEndPreamble{\ifnumcomp{\value{myunitcount}}{=}{1}{\gpreto\@textsupervisor{Thèse préparée dans l'unité de recherche\\}}{\gpreto\@textsupervisor{Thèse préparée dans les unités de recherche\\}}}

	%%%%%%%%%%%%%%%%
	%% TITLE PAGE %%
	%%%%%%%%%%%%%%%%
	\renewcommand{\maketitle}{
		\begin{titlepage}
			\KOMAoption{parskip}{false}
			\begin{spacing}{1}
				\opensans
				\newgeometry{left=6cm,bottom=2cm, top=1cm, right=1cm}
				\begin{tikzpicture}<disable externalization>[remember picture, overlay]
					%\fill[right color=Prune, left color=white] (0,0) rectangle (2,1);
					\node[right color=Prune, left color=white, minimum height=\paperheight, minimum width=7.3mm, shape=rectangle, anchor=south west, inner sep=0pt] (grad) at (current page.south west) {};
					\node[fill=Prune, minimum height=\paperheight, minimum width=27.24mm, shape=rectangle, anchor=south west, inner sep=0pt] (rect) at ([xshift=7.26mm]current page.south west) {};
					\node[anchor=north west, inner sep=0pt] (logo ups) at ([xshift=10.43mm, yshift=-7.1mm] rect.north east) {\includegraphics{logos/Logotype UPSaclay_CMJN.eps}};
					%\node[right= \paperwidth - 107.7mm of logo ups, anchor=east, inner sep=0pt] (logo co) {\includegraphics{logos/Logotype UPSaclay_CMJN.eps}};
					\node[color=white, rotate=90, anchor=north west, inner sep=0] (type) at ([xshift=10.8mm, yshift=16.6mm] current page.south west) {\Large THÈSE DE DOCTORAT};
					\node[right=7.5mm of type.south west, color=white,rotate=90, anchor=north west, inner sep=0] {NNT: 2024UPASG090}; %TODO
				\end{tikzpicture}
				\vspace{3.3cm}
				% logo
				\flushright
				{\LARGE \textcolor{Prune}{\@title}}

				\textit{\@subtitle}

				\vspace{1.7cm}

				\textbf{Thèse de doctorat de l'université Paris-Saclay}

				\vspace{7mm}

				École doctorale n\textdegree\@ednum,\\\@edname~(\@doctoralschool)

				Spécialité de doctorat: \@subject

				Graduate School: \@graduateschool

				Référent: \@referee

				\vspace{7mm}

				\@textsupervisor

				\vspace{15mm}

				\textbf{Thèse soutenue à \@location, le \@date, par}\\\bigskip

				\LARGE{\textcolor{Prune}{\@author}}

				\vspace{\fill}

				\flushleft
				\textcolor{Prune}{\large\textbf{Composition du Jury}}

				\vspace{-0.3\baselineskip}

				\textcolor{Prune}{\small Membres du jury avec voix délibérative}

				\vspace{2mm}

				\@jury
				\restoregeometry
			\end{spacing}
		\end{titlepage}
		\KOMAoption{parskip}{half}
		\cleardoublepage % if print else nothing
	}

	\newcommand{\makeacknolegments}[1]{
		\thispagestyle{plain}
		\addchap{Remerciements}
		\input{#1}
	}

	\newcommand{\makesummary}[2]{
		\KOMAoption{parskip}{false}
		\thispagestyle{empty}
		\newgeometry{left=2cm, right=2cm, top=0.5cm, bottom=1.25cm}
		\begin{spacing}{1}
			\begin{footnotesize}
				\noindent\includegraphics{\@logoed}
				\vfill

				\begin{mdframed}[linecolor=Prune, linewidth=1, innertopmargin=2mm, innerbottommargin=2mm, innerrightmargin=3mm, innerleftmargin=3mm, subtitleinnerbelowskip=1mm, subtitleaboveskip=0mm, subtitlebelowskip=0mm, subtitleinneraboveskip=0mm, skipabove=0mm, skipbelow=0mm]
					\mdfsubtitle{\textcolor{Prune}{Titre}: \@subtitle}
					\noindent\textbf{\textcolor{Prune}{Mots clés}: \@keywordsfr}
					\vspace{-5mm}
					\begin{french}
						\begin{multicols}{2}
							\noindent\textbf{\textcolor{Prune}{\abstractname}}: \input{#1}
						\end{multicols}
					\end{french}

				\end{mdframed}
				\vfill
				\begin{mdframed}[linecolor=Prune, linewidth=1, innertopmargin=2mm, innerbottommargin=2mm, innerrightmargin=3mm, innerleftmargin=3mm, subtitleinnerbelowskip=1mm, subtitleaboveskip=0mm, subtitlebelowskip=0mm, subtitleinneraboveskip=0mm, skipabove=0mm, skipbelow=0mm]
					\mdfsubtitle{\textcolor{Prune}{Title}: \@title}
					\noindent\textbf{\textcolor{Prune}{Keywords}: \@keywordsen}
					\vspace{-5mm}
					\begin{multicols}{2}
						\noindent\textbf{\textcolor{Prune}{\abstractname}}: \input{#2}
					\end{multicols}
				\end{mdframed}
			\end{footnotesize}
			\vfill
		\end{spacing}
		\KOMAoption{parskip}{half}
		\cleardoublepage
		\restoregeometry
	}

	\defbibenvironment{bibnonum}
	{\list
		{}
		{\setlength{\leftmargin}{\bibhang}%
			\setlength{\itemindent}{-\leftmargin}%
			\setlength{\itemsep}{\bibitemsep}%
			\setlength{\parsep}{\bibparsep}}}
	{\endlist}
	{\item}

	\newcommand{\mypublications}{
	\renewbibmacro*{begentry}{\iffieldundef{usera}{}{%
			\llap{\printicon{\thefield{usera}}}\quad}}
	\addchap{Publications}
	\newrefsection[mypubli]
	\nocite{*}
\section*{Accepted}
	\printbibliography[heading=none, env=bibnonum, keyword=submitted]
\section*{Under submission}
	\printbibliography[heading=none, env=bibnonum, keyword=notsubmitted]
	\endrefsection
	}

	\newcommand{\minitocpagetwocols}{
		\begin{multicols}{2}
			\etocsettocstyle{\addsec*{Chapter contents\\\vspace{-15pt}\rule{\textwidth}{0.4pt}\vspace{-10pt}}\KOMAoption{parskip}{false}}{\vspace{-5pt}\noindent\rule{\textwidth}{0.4pt}\clearpage\KOMAoption{parskip}{half}}%
			\localtableofcontents
		\end{multicols}
	}
	\newcommand{\minitocpage}{
		\etocsettocstyle{\addsec*{Chapter contents\\\vspace{-15pt}\rule{\textwidth}{0.4pt}\vspace{-10pt}}\KOMAoption{parskip}{false}\small}{\vspace{-5pt}\noindent\rule{\textwidth}{0.4pt}\clearpage\KOMAoption{parskip}{half}\normalsize}%
		\localtableofcontents
	}
	\newcommand{\minitoc}{
		\etocsettocstyle{\addsec*{Chapter contents\\\vspace{-15pt}\rule{\textwidth}{0.4pt}\vspace{-10pt}}\KOMAoption{parskip}{false}}{\vspace{-5pt}\noindent\rule{\textwidth}{0.4pt}\\\KOMAoption{parskip}{half}\vspace{-20pt}}%
		\localtableofcontents
	}

	\newcommand{\minitoctwocols}{
		\begin{multicols}{2}
			\etocsettocstyle{\vfill\addsec*{Chapter contents\\\vspace{-15pt}\rule{\textwidth}{0.4pt}\vspace{-10pt}}\KOMAoption{parskip}{false}}{\vspace{-5pt}\noindent\rule{\textwidth}{0.4pt}\vfill\clearpage\KOMAoption{parskip}{half}}%
			\localtableofcontents
		\end{multicols}
	}

	\newcommand{\minitocpagecentered}{
		\etocsettocstyle{\vfill\addsec*{Chapter contents\\\vspace{-15pt}\rule{\textwidth}{0.4pt}\vspace{-10pt}}\KOMAoption{parskip}{false}}{\vspace{-5pt}\noindent\rule{\textwidth}{0.4pt}\vfill\clearpage\KOMAoption{parskip}{half}}%
		\localtableofcontents
	}
	% almost there...
	% https://tex.stackexchange.com/questions/230162/what-is-the-name-of-this-page-layout-parameter-in-koma-script
	%https://tex.stackexchange.com/questions/423091/chapter-heading-designs-in-koma-script
	%https://github.com/derric/cleanthesis/tree/master
	%https://tex.stackexchange.com/questions/485065/accessing-chapters-name-in-koma-script

	%% CUSTOMIZE \bibbysection to exclude my publications
	\RequirePackage{refcount}
	\def\blx@refsections{%
		\ifcsvoid{blx@dlist@entry@\the\c@refsection @\blx@refcontext@context}
		{}
		{\toggletrue{blx@tempa}%
			\protected@edef\@tempa{\getrefbykeydefault{refsection:\the\c@refsection}{name}{}}
			\ifdefstring{\@tempa}{Publications}{}{
				\begingroup
				\expandafter\blx@bibliography\csname blx@dlist@entry@\the\c@refsection @\blx@refcontext@context\endcsname}%
		}

		\ifnum\c@refsection<\c@blx@maxsection
			\advance\c@refsection\@ne
			\expandafter\blx@refsections
		\else
			\iftoggle{blx@tempa}{}{\blx@warn@bibempty}%
			\endgroup
		\fi}

	%% CUSTOMIZE CHAPTER TITLE PAGE
	\RedeclareSectionCommand[afterskip=0.8\baselineskip plus 0.115\baselineskip minus 0.192\baselineskip]{chapter}
	\newcommand\HUGE{\fontsize{60}{0}\selectfont}

	\renewcommand*\chapterformat{%
		\smash{\textcolor{Prune}{\rule[-5pt]{2pt}{10cm}}}%
		\quad
		\textcolor{Prune}{\HUGE\thechapter}%
	}%

	\renewcommand*\chapterlinesformat[3]{%
		\IfArgIsEmpty{#2}{
			\@hangfrom{#2}{#3}
		}{
			\vspace*{-4mm}
			\parbox[t]{\textwidth}{\raggedchapter #3}%
			\hfill
			\makebox[0pt][l]{#2}% l in margin
		}
	}

	\DeclareNewLayer[
		background,
		bottommargin,
		%addvoffset=1.4cm,
		addhoffset=\hoffset+1in+\oddsidemargin,
		width=\textwidth,
		mode=text,
		oddpage,
		contents=\vfill\hfill Compiled on \DTMToday\xspace from \href{https://github.com/abeaude/thesis/tree/\gitHash}{[\gitBranch] -- \gitAbbrevHash}\xspace\textcolor{Prune}{\gitDirty}\hfill\vfill
	]{gitmetadata}
	\if@gitmark \AddLayersToPageStyle{scrheadings}{gitmetadata} \fi

	%% DEFINE HEADER AND FOOTER %%
	\clearpairofpagestyles
	\renewcommand*{\chaptermarkformat}{}
	\renewcommand*{\sectionmarkformat}{}
	\automark[section]{chapter}
	\ohead{\leftmark}
	\chead{}
	\ihead{\rightmark}
	\ifoot*{}

	\if@twoside%
		\cfoot*{}
		\ofoot*{\pagemark}
		% \lefoot*{
		% 	\begin{tikzpicture}<disable externalization>
		% 		\node (b) [inner sep=0pt]{\pagemark};
		% 		\node[inner sep=0pt, right=1mm of b.east, anchor=west]{\pgfornament[height=2ex, symmetry=v]{72}};
		% 	\end{tikzpicture}
		% }
		% \rofoot*{
		% 	\begin{tikzpicture}<disable externalization>
		% 		\node (a) [inner sep=0pt]{\pgfornament[height=2ex]{72}};
		% 		\node [right=1mm of a.east, inner sep=0pt, anchor=west]{\pagemark};
		% 	\end{tikzpicture}
		% }
	\else%
		\cfoot*{\begin{tikzpicture}<disable externalization>
				\node (a) [inner sep=0pt]{\pgfornament[height=2ex]{72}};
				\node (b) [right=1mm of a.east, inner sep=0pt, anchor=west]{\pagemark};
				\node[inner sep=0pt, right=1mm of b.east, anchor=west]{\pgfornament[height=2ex, symmetry=v]{72}};
			\end{tikzpicture}}
		%  \cfoot*{\pagemark}
		\ofoot*{}
	\fi%

	\newpairofpagestyles
	[scrheadings]% scrheadings as parent style -> clone its *current* settings
	{appendix}% name of the new layer page style
	{
		\automark[chapter]{part}
		\ohead{\leftmark}
		\ihead{\rightmark}
	}
	\ExplSyntaxOn
	\NewDocumentCommand{\MyrobExtCopyFileToCache}{mm}{
	\robExtCheckIfPrefixFolderExists%
	\file_if_exist:xTF {#1}{}{
		\msg_error:nnx{robExt}{file does not exist}{#1~(to~copy~to~cache)}
	}
	\file_if_exist:xTF {\robExtAddCachePath{#2}}{
	\robExtDebugInfo{The~file~\robExtAddCachePath{#2}~already~exists,~let~us~compare~md5~sum.}
	\str_set:Nx \l_tmpa_str {\file_mdfive_hash:n{#1}}
	\str_set:Nx \l_tmpb_str {\file_mdfive_hash:n{\robExtAddCachePath{#2}}}
	\str_compare:eNeTF {\l_tmpa_str} = {\l_tmpb_str} {
	\robExtDebugInfo{The~file~\robExtAddCachePath{#2}~has~the~good~md5~hash.}
	}{
	\robExtDebugInfo{The~md5~hashes~of~#1~and~\robExtAddCachePath{#2}~are~different(\l_tmpa_str \space vs \space \l_tmpb_str),~let~us~copy~the~file.}%
	\sys_shell_now:x {\robExtPrefixAllCompilationCommands \robExtCp \space "#1"~"\robExtAddCachePath{#2}"}% 
	}
	}{
	\robExtDebugInfo{The~file~\robExtAddCachePath{#2}~does~not~exist,~let~us~copy~it:}
	\sys_shell_now:x {\robExtPrefixAllCompilationCommands \robExtCp \space "#1"~"\robExtAddCachePath{#2}"}%    
	}
	}
	\NewDocumentCommand{\CreateCacheDirs}{}{\sys_shell_now:x {mkdir ~ -p ~ \robExtOutputDirectory / \robExtCacheFolder /fonts}}
	\prop_new:N \l_answerin_icons_prop
	\prop_set_from_keyval:Nn \l_answerin_icons_prop {
		book = {\faBook},
		article = {\faFile*},
		pres = {\faSlideshare}
	}

	\NewDocumentCommand{\printicon}{m}{
		\prop_item:Ne \l_answerin_icons_prop {#1}
	}
	\ExplSyntaxOff
	\NewCommandCopy{\orgdescriptionlabel}{\descriptionlabel}
	\NewDocumentCommand{\PrintChapterBiblioAndPubli}{}{
		\KOMAoption{parskip}{false}
		\begin{singlespace}
			%%%%%%%%%%%%%%%%%%
			%% PUBLICATIONS %%
			%%%%%%%%%%%%%%%%%%
			\mypublications

			%%%%%%%%%%%%%%%%%
			%% BIBLIOGRAPY %%
			%%%%%%%%%%%%%%%%%
			\renewbibmacro*{begentry}{}
			\printbibheading
			\bibbysection[heading=subbibliography]
			\printglossary[title=Glossary]
		\end{singlespace}
		\KOMAoption{parskip}{half}
	}
	\gappto\appendix{%
		\captionsetup[figure]{list=no}%
		\captionsetup[table]{list=no}%
		\crefalias{figure}{appendixfigure}%
		\crefalias{subfigure}{appendixfigure}%
		\crefalias{table}{appendixtable}%
		\renewcaptionname{english}{\figurename}{Appendix figure}%
		\renewcaptionname{english}{\tablename}{Appendix table}%
		\pagestyle{appendix}%
	}
	\gappto\mainmatter{%
		% Redefinition of Description List Items source: https://tex.stackexchange.com/questions/695120/error-when-combining-glossaries-and-label-of-description-item
		\renewcommand*{\descriptionlabel}[1]{%
			\phantomsection
			\begingroup\let\label\@gobble\protected@edef\x{\endgroup
				\def\noexpand\@currentlabel{#1}%
				\def\noexpand\@currentlabelname{#1}%
			}\x
			\orgdescriptionlabel{#1}%
		}
	}