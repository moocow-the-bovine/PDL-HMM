#!/usr/bin/perl -wd

use lib qw(./blib/lib ./blib/arch);
use PDL;
use PDL::HMM;

BEGIN{ $, = ' '; }

##---------------------------------------------------------------------
## log addition

##-- addlogs($x,$y): wrapper for ($x+$y) via logadd()
sub addlogs { return exp(logadd(map { log(pdl($_)) } @_)); }

##-- addlogs1(@values): wrapper for (sum(@values)) via logsumover()
sub addlogs1 { return exp(logsumover(pdl(double,\@_)->log)); }

##-- naive case
sub logadd0  { my $sum=0; $sum+=exp($_) foreach (@_); return log($sum); }
sub addlogs0 { return exp(logadd0(map { log(pdl($_)) } @_)); }



##-- subtraction
sub logdiff0 { my ($x,$y)=map {exp($_)} @_; return log($x>$y ? $x-$y : $y-$x); }
sub sublogs0 { return exp(logdiff0(map {log($_)} @_)); }

sub logdiff1b {
  my $xy=pdl(double, \@_);
  my $x=$xy->slice("0");
  my $y=$xy->slice("1");
  return maximum($xy) + log(exp($x-maximum($xy)) - exp($y-maximum($xy)));
}
sub sublogs11b { return exp(logdiff1b(map {log($_)} @_)); }

BEGIN { $LOG_BIG = 72; }
sub logdiff1c {
  my $xy=pdl(double, \@_);
  my $x=$xy->slice("0");
  my $y=$xy->slice("1");
  if    ($y-$x > $LOG_BIG) { print "y-x > BIG!\n"; return $y; }
  elsif ($x-$y > $LOG_BIG) { print "x=$x-y=$y > BIG!\n"; return $x; }
  return maximum($xy) + log(exp($x-maximum($xy)) - exp($y-maximum($xy)));
}
sub sublogs11c { return exp(logdiff1c(map {log($_)} @_)); }


sub logdiff1 { return PDL::HMM::logdiff(map { pdl(double,$_) } @_); }
sub sublogs1 { return exp(logdiff1(map {log($_)} @_)); }



##---------------------------------------------------------------------
## text probability

use vars qw($fwp $fwfp);

## normalization
sub normpdl {
  my $ppdl = shift;
  return $ppdl /= $ppdl->xchg(0,1)->sumover;
}

sub model1 {
  $n = 2;
  $k = 2;

  ##-- a
  $af = zeroes(double, $n,$n);
  $af->slice("(0)") .= pdl([2,2]);    ## p(0-->0)=.5 ; p(0-->1)=.5
  $af->slice("(1)") .= pdl([2,2]);    ## p(1-->0)=.5 ; p(1-->1)=.5

  ##-- b
  $bf = zeroes(double, $n,$k);
  $bf->slice("(0)") .= pdl([1,0]);    ## p(0 @ 0)=.75 ; p(1 @ 0)=0   ; p(2 @ 0)=.25
  $bf->slice("(1)") .= pdl([0,1]);    ## p(0 @ 1)=0   ; p(1 @ 1)=.75 ; p(2 @ 1)=.25
  $b = log(normpdl($bf));
  $bp = $b->exp;

  ##-- pi
  $pif = pdl(double,[1,1]);         ## initial probability: uniform
  $pi = log($pif/$pif->sum);
  $pip = $pi->exp;

  ##-- omega
  $omegaf = pdl(double,[1,1]);      ## final probability: uniform

  ##-- sanity: a+omega==1
  ##-- plain
  #$afsumover = $af->xchg(0,1)->sumover + $omegaf;
  #$a = log($af / $afsumover);
  #$omega = log($omegaf / $afsumover);

  ##-- simulate maximize()
  $logafsumover = log($af)->xchg(0,1)->logsumover->inplace->logadd(log($omegaf));
  $a = log($af) - $logafsumover;
  $omega = log($omegaf) - $logafsumover;

  $ap = $a->exp;
  $omegap = $omega->exp;
}

sub tptest1 {
  model1();

  ##-- o
  $o = pdl([0,1]);

  ##-- alpha
  $fw = hmmfw($a,$b,$pi, $o);
  $fwp = exp($fw);

  $fwp_expect  = pdl(double, [[1/2,0],[0,1/5]]);
  print "fw:  ", (all($fwp->approx($fwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n";

  ##-- beta
  $bw  = hmmbw($a,$b,$omega, $o);
  $bwp = exp($bw);

  $bwp_expect  = pdl(double, [[2/25,2/25],[1/5,1/5]]);
  print "bw:  ", (all($bwp->approx($bwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n";

  ##-- text prob
  $fwtp = sumover($fwp->slice(",-1") * $omegap);
  $bwtp = sumover($bwp->slice(",0")  * $pip * $bp->slice(",(".$o->at(0).")"));
  print "fw==bw ? ", (all($fwtp->approx($bwtp),1e-6) ? "OK" : "NOT OK"), "\n";
}


##---------------------------------------------------------------------
## Re-estimation
sub emtest1i {
  model1();

  ##-- os
  @os = ((map { pdl([0,1]) } (1..4)),  (map { pdl([1,0]) } (1..2)));

  emtest1e0();
}

sub emtest1e0 {
  ##-- re-est: base
  ($ea,$eb,$epi,$eomega) = hmmexpect0($a,$b,$pi,$omega);
  $etp = logzero;
}

use vars qw($eaf $ebf $epif $eomegaf);
sub emtest1e {
  foreach $o (@os) {
    ##-- alpha
    $fw = hmmfw($a,$b,$pi, $o); $fwp = exp($fw);

    ##-- beta
    $bw  = hmmbw($a,$b,$omega, $o); $bwp = exp($bw);

    ##-- text-prob
    $etp->inplace->logadd(logsumover($fw->slice(",-1") + $omega));

    ##-- re-est: expect
    hmmexpect($a,$b,$pi,$omega, $o, $fw,$bw, $ea,$eb,$epi,$eomega);
  }
  $etp -= log(scalar(@os));
  foreach $o (@os) {
    ##-- alpha
    $fw = hmmfw($a,$b,$pi, $o); $fwp = exp($fw);

    ##-- beta
    $bw  = hmmbw($a,$b,$omega, $o); $bwp = exp($bw);

    ##-- text-prob
    $etp->inplace->logadd(logsumover($fw->slice(",-1") + $omega));

    ##-- re-est: expect
    hmmexpect($a,$b,$pi,$omega, $o, $fw,$bw, $ea,$eb,$epi,$eomega);
  }
  $etp -= log(scalar(@os));

  $eaf = $ea->exp;
  $ebf = $eb->exp;
  $epif = $epi->exp;
  $eomegaf = $eomega->exp;
}

use vars qw($ahatp $bhatp $pihatp $etpdiff $etpdiffp);
sub emtest1m {
  ##-- re-est: maximimize
  ($ahat,$bhat,$pihat,$omegahat) = hmmmaximize($ea,$eb,$epi,$eomega);

  ##-- re-est: get new textprob
  $etphat = logzero;
  foreach $o (@os) {
    ##-- alpha
    $fw = hmmfw($ahat,$bhat,$pihat, $o);
    $etphat->inplace->logadd(logsumover($fw->slice(",-1") + $omegahat));
  }
  $etphat -= log(scalar(@os));

  ##-- now can compare $etphat <-> $etp
  $etpdiff = $etphat->logdiff($etp);

  ##-- get exps
  ($ahatp,$bhatp,$pihatp,$omegahatp,$etpdiffp) = map { exp($_) } ($ahat,$bhat,$pihat,$omegahat,$etpdiff);
}

sub emtest1u {
  ($a,$b,$pi,$omega, $ap,$bp,$pip,$omegap) = ($ahat,$bhat,$pihat,$omegahat, $ahatp,$bhatp,$pihatp,$omegahatp);
  emtest1e0;
}


##---------------------------------------------------------------------
## Viterbi algorithm
sub vtest1 {
  $n = 2; $k = 2;

  $pi = pdl(double,[.6,.4])->log;

  $a = pdl(double,   [[.5, .2],
		      [.3, .5]])->log;
  $omega = pdl(double,[.2, .3])->log;

  $b = pdl(double, [[.8, .2], [.2, .8]])->log;

  ##-- aliases
  ($ap,$bp,$pip,$omegap) = map { exp($_) } ($a,$b,$pi,$omega);

  ##-- o
  $o = pdl([0,1]); $t=$o->nelem;
}

use vars qw($delta $deltap $psi $t $path $best);
sub vtest2 {
  #hmmviterbi($a,$b,$pi, $o, ($delta=zeroes(double,$n,$t)), ($psi=zeroes(long,$n,$t)));
  ##--
  #($delta,$psi) = hmmviterbi($a,$b,$pi,$omega, $o);  $deltap = $delta->exp;
  ($delta,$psi) = hmmviterbi($a,$b,$pi, $o);  $deltap = $delta->exp;

  $path = hmmpath($psi, sequence($n));
  $qbest = maximum_ind($delta->slice(",-1") + $omega)->at(0);

  use vars qw($pbest);
  $pbest = $path->slice(",($qbest)");
}

##---------------------------------------------------------------------
## DUMMY
##---------------------------------------------------------------------
foreach $i (0..100) {
  print "--dummy($i)--\n";
}

