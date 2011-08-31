use PDL::Core::Dev;
use ExtUtils::MakeMaker;
PDL::Core::Dev->import();

$package  = ["HMM.pd",HMM,PDL::HMM];
%hash = pdlpp_stdargs($package);

$hash{AUTHOR} = 'Bryan Jurish';
$hash{ABSTRACT} = 'PDL Hidden Markov Model utilities';
$hash{PREREQ_PM}{PDL} = '2.4.2'; ##-- ensure PDL-ability
push(@{$hash{LIBS}}, '-lm');
#$hash{INC} .= '';
$hash{DIR} = [];
#$hash{OBJECT} .= '';
$hash{VERSION_FROM} = 'HMM.pd';
$hash{realclean}{FILES} .= ' *~ *.tmp README.txt';

##--
## hacks for CPAN indexer
##   + include PODs from PDL::PP-generated *.pm file to get docs on CPAN
##   + BUT unlink it here so that its presence doesn't clobber
##     the PDL::PP rules in postamble
$pkg = $package->[1]; ##-- package basename
unlink "$pkg.pm" if (-e "$pkg.pm");

WriteMakefile(%hash);
print STDERR "done\n";

sub MY::depend {
  package MY;
  return shift->SUPER::depend(@_) .<<EOF;

dist: README.txt $::pkg.pm

distcheck: README.txt $::pkg.pm

create_distdir: README.txt $::pkg.pm

EOF
}

sub MY::special_targets {
  package MY;
  return shift->SUPER::special_targets(@_) .<<EOF;

.SUFFIXES: .pm .pod .man .txt .html

.pm.html:
	pod2html --outfile \$@ \$<

.pm.txt:
	pod2text \$< \$@

.pod.html:
	pod2html --outfile \$@ \$<

.pod.txt: README.pod
	pod2text \$< \$@

EOF
}

sub MY::distdir {
  package MY;
  my $inherited = shift->SUPER::distdir(@_);
  $inherited =~ s/^(create_distdir\s*:.*\R(?:\t.*\R)*)/$1\t\$(CP) $::pkg.pm \$(DISTVNAME)\n/m;
  $inherited;
}

#sub MY::top_targets {
#  package MY;
#  my $inherited = shift->SUPER::top_targets(@_);
#  $inherited;
#}

sub MY::postamble {
  pdlpp_postamble($package);
}
