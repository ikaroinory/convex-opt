%----------------------------------------------------------------------------------------------------%
%
% -> Class Clarification
%
%----------------------------------------------------------------------------------------------------%
\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{reportx}[2024/11/15 v0.1 Report template for student in Harbin Institute of Technology]

\LoadClass[zihao = -4]{ctexart}

%----------------------------------------------------------------------------------------------------%
%
% -> Package
%
%----------------------------------------------------------------------------------------------------%
% ----- Control -----
\RequirePackage{ifthen}
\RequirePackage{ifplatform}
\RequirePackage{xpatch}

% ----- Layout -----
\RequirePackage[a4paper,hmargin=3.17cm,vmargin=2.54cm]{geometry}

% ----- Mathematics -----
\RequirePackage{amsmath}
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

% ----- Bibliography -----
\RequirePackage{gbt7714}

% ----- Caption -----
\RequirePackage{caption}
\RequirePackage{subcaption}

% ----- Table -----
\RequirePackage{booktabs}

% ----- Images -----
\RequirePackage{graphicx}

% ----- Footer -----
\RequirePackage{fancyhdr}

%----------------------------------------------------------------------------------------------------%
%
% -> Font
%
%----------------------------------------------------------------------------------------------------%
\ExplSyntaxOn
    \xpatchcmd\__xeCJK_check_family:n{\__xeCJK_warning:nxx}{\__xeCJK_info:nxx}{}{}
\ExplSyntaxOff

\ifwindows
    \setCJKmainfont[AutoFakeBold, AutoFakeSlant]{SimSun}
\fi
\ifmacosx
    \setCJKmainfont[AutoFakeBold, AutoFakeSlant]{Songti SC}
\fi

%----------------------------------------------------------------------------------------------------%
%
% -> Text Format
%
%----------------------------------------------------------------------------------------------------%
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexart}}
\ProcessOptions\relax

%----------------------------------------------------------------------------------------------------%
%
% -> Math Formula
%
%----------------------------------------------------------------------------------------------------%
\renewcommand*{\baselinestretch}{1.5}
\setlength{\baselineskip}{16pt}
\setlength{\parskip}{12pt}

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
% -> Reference
%
%----------------------------------------------------------------------------------------------------%
\AtBeginDocument{
    \crefformat{table}{#2表 #1#3 }
    \crefformat{figure}{#2图 #1#3 }
    \crefformat{subfigure}{#2图 #1#3 }
    \crefformat{equation}{#2公式 (#1)#3 }
    \crefformat{algorithm}{#2算法 #1#3 }
    \crefformat{subsection}{ #2#1 小节#3}

    \crefrangeformat{equation}{公式(#3#1#4) $\sim$ (#5#2#6)}
    \crefrangeformat{figure}{#3图 #1#4至#5图 #2#6 }

    \crefmultiformat{figure}{#2图 #1#3}{, #2图 #1#3 }{, #2图 #1#3 }{和#2图 #1#3 }
}

%----------------------------------------------------------------------------------------------------%
%
% -> Bibliography
%
%----------------------------------------------------------------------------------------------------%
\bibliographystyle{gbt7714-numerical}

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
    font            =   { sfive },
    justification   =   centering
}

%----------------------------------------------------------------------------------------------------%
%
% -> Footer
%
%----------------------------------------------------------------------------------------------------%
\pagestyle{plain}

\RequirePackage{titlesec}

\titleformat{\section}{\songti \bf \zihao{-3}}{\chinese{section}、}{0em}{}

\titleformat{\subsection}{\songti \bf \zihao{4}}{\thesubsection}{1em}{}

\titleformat{\subsubsection}{\songti \bf \zihao{-4}}{\thesubsubsection}{1em}{}
