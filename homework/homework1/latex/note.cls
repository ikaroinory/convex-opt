%----------------------------------------------------------------------------------------------------%
%
% -> Class Clarification
%
%----------------------------------------------------------------------------------------------------%
\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{hitrep}[2024/11/15 v0.1 Report template for student in Harbin Institute of Technology]

\LoadClass[zihao = -4]{ctexart}

%----------------------------------------------------------------------------------------------------%
%
% -> Package
%
%----------------------------------------------------------------------------------------------------%
% ----- Control -----
\RequirePackage{ifthen}
\RequirePackage{ifplatform}

% ----- Layout -----
\RequirePackage[a4paper,hmargin=2.54cm,vmargin=2.54cm]{geometry}
\usepackage{setspace}

% ----- Fix font size -----
\RequirePackage{anyfontsize}

% ----- Color -----
\RequirePackage{color}
\RequirePackage[dvipsnames, svgnames, x11names]{xcolor}

% ----- Enum list -----
\RequirePackage{enumitem}

% ----- Math -----
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amsfonts}
\RequirePackage{bm}

% ----- Links -----
\RequirePackage[
    hidelinks,
    breaklinks = true,
    bookmarksopen = true,
    bookmarksopenlevel = 2
]{hyperref}

% ----- References -----
\RequirePackage[nameinlink]{cleveref}

% ----- Float element -----
\RequirePackage{float}
\RequirePackage{caption}

% ----- Table -----
\RequirePackage{booktabs}
\RequirePackage{makecell}
\RequirePackage{multirow}
\RequirePackage{colortbl}
\RequirePackage{longtable}

% ----- Images -----
\RequirePackage{graphicx}
\RequirePackage{subcaption}

% ----- Algorithms -----
\RequirePackage{algorithm}
\RequirePackage{algorithmicx}
\RequirePackage[noend]{algpseudocode}

%----------------------------------------------------------------------------------------------------%
%
% -> Options
%
%----------------------------------------------------------------------------------------------------%
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexart}}
\ProcessOptions\relax

%----------------------------------------------------------------------------------------------------%
%
% -> Global Variables
%
%----------------------------------------------------------------------------------------------------%
\gdef\hitrep@subject{}
\NewDocumentCommand{\subject}{m}{
    \gdef\hitrep@subject{#1}
}
\gdef\hitrep@department{}
\NewDocumentCommand{\department}{m}{
    \gdef\hitrep@department{#1}
}
\gdef\hitrep@major{}
\NewDocumentCommand{\major}{m}{
    \gdef\hitrep@major{#1}
}
\gdef\hitrep@author{}
\RenewDocumentCommand{\author}{m}{
    \gdef\hitrep@author{#1}
}
\gdef\hitrep@studentid{}
\NewDocumentCommand{\studentid}{m}{
    \gdef\hitrep@studentid{#1}
}
\gdef\hitrep@studenttype{}
\NewDocumentCommand{\studenttype}{m}{
    \gdef\hitrep@studenttype{#1}
}

%----------------------------------------------------------------------------------------------------%
%
% -> Table of Content
%
%----------------------------------------------------------------------------------------------------%
\newcommand{\toc}{
    \phantomsection
    \addcontentsline{toc}{section}{\contentsname}
    \tableofcontents
}

%----------------------------------------------------------------------------------------------------%
%
% -> Fonts
%
%----------------------------------------------------------------------------------------------------%
\setmainfont{Times New Roman}

\ifwindows
    \setCJKmainfont[AutoFakeBold, AutoFakeSlant]{SimSun}
\fi
\ifmacosx
    \setCJKmainfont[AutoFakeBold, AutoFakeSlant]{Songti SC}
\fi

%----------------------------------------------------------------------------------------------------%
%
% -> Enum List
%
%----------------------------------------------------------------------------------------------------%
\setlength{\itemindent}{0pt}

%----------------------------------------------------------------------------------------------------%
%
% -> Algorithms
%
%----------------------------------------------------------------------------------------------------%
\floatname{algorithm}{算法}

\renewcommand{\algorithmicrequire}{\textbf{输入：}}
\renewcommand{\algorithmicensure}{\textbf{输出：}}

\newcommand{\Break}{\State\textbf{break}}
\renewcommand{\Call}[2]{#1(#2)}
\newcommand{\Get}{ $\gets$ }

\algrenewtext{Function}[2]{\textbf{function} #1(#2)}

%----------------------------------------------------------------------------------------------------%
%
% -> Caption
%
%----------------------------------------------------------------------------------------------------%
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{sfive}{\zihao{5}}

\captionsetup{
    format          =   plain,
    labelsep        =   quad,
    font            =   { song, sfive },
    justification   =   centering
}

\AtBeginDocument{
    \crefformat{table}{#2表 #1#3 }
    \crefformat{figure}{#2图 #1#3 }
    \crefformat{subfigure}{#2图 #1#3 }
    \crefformat{equation}{#2公式 (#1)#3 }
    \crefformat{algorithm}{#2算法 #1#3 }
    \crefformat{subsection}{ #2#1 小节#3}

    \crefrangeformat{equation}{公式(#3#1#4) $\sim$ (#5#2#6)}
}

%----------------------------------------------------------------------------------------------------%
%
% -> Title Style
%
%----------------------------------------------------------------------------------------------------%
\RequirePackage{titlesec}

\titleformat{\section}{\songti \bf \zihao{-3}}{\thesection}{1em}{}

\titleformat{\subsection}{\songti \bf \zihao{4}}{\thesubsection}{1em}{}

\titleformat{\subsubsection}{\songti \bf \zihao{-4}}{\thesubsubsection}{1em}{}

%----------------------------------------------------------------------------------------------------%
%
% -> Math Formula
%
%----------------------------------------------------------------------------------------------------%
\everymath{\displaystyle}
\setlength{\abovedisplayskip}{0pt}
\setlength{\belowdisplayskip}{0pt}

%----------------------------------------------------------------------------------------------------%
%
% -> References
%
%----------------------------------------------------------------------------------------------------%
\RequirePackage{gbt7714}

\bibliographystyle{gbt7714-numerical}

\newcommand{\bib}[1]{
    \phantomsection
    \addcontentsline{toc}{section}{\refname}
    \bibliography{#1}
}

%----------------------------------------------------------------------------------------------------%
%
% -> New Environments
%
%----------------------------------------------------------------------------------------------------%
\newcounter{hitrep@counter@numpar}
\def\hitrep@oldcmd@item{\item}
\newenvironment{numpar}{%
    \def\item{\par\refstepcounter{hitrep@counter@numpar}(\thehitrep@counter@numpar) }%
}{%
    \setcounter{hitrep@counter@numpar}{0}%
    \def\item\hitrep@oldcmd@item%
}

%----------------------------------------------------------------------------------------------------%
%
% -> Table
%
%----------------------------------------------------------------------------------------------------%
\let\oldtabular\tabular
\let\endoldtabular\endtabular
\renewenvironment{tabular} {
    \bgroup
	\renewcommand{\arraystretch}{1.38}
    % \setlength\tabcolsep{9pt}
    \oldtabular
} {
    \endoldtabular\egroup
}
