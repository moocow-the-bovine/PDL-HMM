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

sub tptest1 {
  $n = 3; $k = 3;

  ##-- a
  $ap = ones(double, $n,$n);  ## uniform arc probability
  $ap->set(0,0, 0);           ## p(0-->0) = 0
  $a = log(normpdl($ap));

  ##-- b
  $bp = ones(double, $n,$k);  ## uniform observation probability
  $bp->slice("0,1:2") .= 0;   ## p(O!=0 @ S==0) = 0
  $bp->slice("1:2,0") .= 0;   ## p(O==0 @ S!=0) = 0
  $b = log(normpdl($bp));

  ##-- pi
  $pip = pdl(double,[1,0,0]); ## initial probability: bos
  $pi = log($pip/$pip->sum);

  ##-- o
  $o = pdl([0,1,2,0]);

  ##-- alpha
  $fw  = hmmfw($a,$b,$pi,$o);
  $fwp = exp($fw);


  $fwp_expect  = pdl(double, [[1,0,0], [0,1/4,1/4], [0,1/12,1/12], [1/18,0,0]]);
  print "fw:  ", (all($fwp->approx($fwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n";

  ##-- beta
  $bw  = hmmbw($a,$b,$o);
  $bwp = exp($bw);

  $bwp_expect  = pdl(double, [[1/18,1/27,1/27], [1/6,1/9,1/9], [0,1/3,1/3], [1,1,1]]);
  print "bw:  ", (all($bwp->approx($bwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n";

  ##-- text prob
  $fwtp = $fwp->slice(",".($fw->dim(1)-1))->sumover;
  $bwtp = sumover($pip*$bwp->slice(",0"));
  print "fw==bw ? ", (all($fwtp->approx($bwtp),1e-6) ? "OK" : "NOT OK"), "\n";
}


sub tptest2 {
  $n = 3;
  $k = 3;

  ##-- a
  $ap = zeroes(double, $n,$n);
  $ap->slice("(0)") .= pdl([0,1,1]);  ## p(0-->0)=0   ; p(0-->1)=0.5 ; p(0-->2)=0.5
  $ap->slice("(1)") .= pdl([1,0,1]);  ## p(1-->0)=0.5 ; p(1-->1)=0.5 ; p(1-->2)=0.5
  $ap->slice("(2)") .= pdl([1,1,0]);  ## p(2-->0)=0.5 ; p(2-->1)=0.5 ; p(2-->2)=0
  $a = log(normpdl($ap));

  ##-- b
  $bp = zeroes(double, $n,$k);
  $bp->slice("(0)") .= pdl([1,0,0]);  ## p(0 @ 0)=1   ; p(1 @ 0)=0   ; p(2 @ 0)=0
  $bp->slice("(1)") .= pdl([0,1,0]);  ## p(0 @ 1)=0   ; p(1 @ 1)=1   ; p(2 @ 1)=0
  $bp->slice("(2)") .= pdl([0,0,1]);  ## p(0 @ 2)=0   ; p(1 @ 2)=0   ; p(2 @ 2)=1
  $b = log(normpdl($bp));

  ##-- pi
  $pip = pdl(double,[1,0,0]);      ## initial probability: bos
  $pi = log($pip/$pip->sum);

  ##-- o
  $o = pdl([0,1,2,0]);

  ##-- alpha
  $fw = hmmfw($a,$b,$pi, $o);
  $fwp = exp($fw);

  $fwp_expect  = pdl(double, [[1,0,0], [0,1/2,0], [0,0,1/4], [1/8,0,0]]);
  print "fw:  ", (all($fwp->approx($fwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n";


  ##-- beta
  $bw  = hmmbw($a,$b,$o);
  $bwp = exp($bw);

  $bwp_expect  = pdl(double, [[1/8,0,1/8], [1/4,1/4,0], [0,1/2,1/2], [1,1,1]]);
  print "bw:  ", (all($bwp->approx($bwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n";

  ##-- text prob
  $fwtp = $fwp->slice(",".($fw->dim(1)-1))->sumover;
  $bwtp = sumover($pip*$bwp->slice(",0"));
  print "fw==bw ? ", (all($fwtp->approx($bwtp),1e-6) ? "OK" : "NOT OK"), "\n";
}


##---------------------------------------------------------------------
## Re-estimation
sub emtest1i {
  $n = 3;
  $k = 3;

  ##-- a
  $ap = zeroes(double, $n,$n);
  $ap->slice("(0)") .= pdl([0,1,1]);  ## p(0-->0)=0   ; p(0-->1)=0.5 ; p(0-->2)=0.5
  $ap->slice("(1)") .= pdl([1,0,1]);  ## p(1-->0)=0.5 ; p(1-->1)=0.5 ; p(1-->2)=0.5
  $ap->slice("(2)") .= pdl([1,1,0]);  ## p(2-->0)=0.5 ; p(2-->1)=0.5 ; p(2-->2)=0
  $a = log(normpdl($ap));

  ##-- b
  $bp = zeroes(double, $n,$k);
  $bp->slice("(0)") .= pdl([1,0,0]);  ## p(0 @ 0)=1   ; p(1 @ 0)=0   ; p(2 @ 0)=0
  $bp->slice("(1)") .= pdl([0,1,0]);  ## p(0 @ 1)=0   ; p(1 @ 1)=1   ; p(2 @ 1)=0
  $bp->slice("(2)") .= pdl([0,0,1]);  ## p(0 @ 2)=0   ; p(1 @ 2)=0   ; p(2 @ 2)=1
  $b = log(normpdl($bp));

  ##-- pi
  $pip = pdl(double,[1,0,0]);     ## initial probability: bos
  $pi = log($pip/$pip->sum);

  ##-- os
  @os = ((map { pdl([0,1,2,0]) } (1..4)),  (map { pdl([0,2,1,0]) } (1..2)));

  emtest1e0();
}

sub emtest1e0 {
  ##-- re-est: base
  ($ea,$eb,$epi) = hmmexpect0($a,$b,$pi);
  $etp           = logzero;
}

use vars qw($eaf $ebf $epif);
sub emtest1e {
  foreach $o (@os) {
    ##-- alpha
    $fw = hmmfw($a,$b,$pi,$o); $fwp = exp($fw);

    ##-- beta
    $bw  = hmmbw($a,$b,$o); $bwp = exp($bw);

    ##-- text-prob
    $etp->inplace->logadd(logsumover($fw->slice(",".($fw->dim(1)-1))));

    ##-- re-est: expect
    hmmexpect($a,$b,$pi, $o, $fw,$bw, $ea,$eb,$epi);
  }
  $etp -= log(scalar(@os));

  $eaf = $ea->exp;
  $ebf = $eb->exp;
  $epif = $epi->exp;
}

use vars qw($ahatp $bhatp $pihatp $etpdiff $etpdiffp);
sub emtest1m {
  ##-- re-est: maximimize
  ($ahat,$bhat,$pihat) = hmmmaximize($ea,$eb,$epi);

  ##-- re-est: get new textprob
  $etphat = logzero;
  foreach $o (@os) {
    ##-- alpha
    $fw = hmmfw($ahat,$bhat,$pihat, $o);
    $etphat->inplace->logadd(logsumover($fw->slice(",".($fw->dim(1)-1))));
  }
  $etphat -= log(scalar(@os));

  ##-- now can compare $etphat <-> $etp
  $etpdiff = $etphat->logdiff($etp);

  ##-- get exps
  ($ahatp,$bhatp,$pihatp,$etpdiffp) = map { exp($_) } ($ahat,$bhat,$pihat,$etpdiff);
}

sub emtest1u {
  ($a,$b,$pi, $ap,$bp,$pip) = ($ahat,$bhat,$pihat, $ahatp,$bhatp,$pihatp);
  emtest1e0;
}


##---------------------------------------------------------------------
## Viterbi algorithm
sub vtest1 {
  $n = 3; $k = 3;

  ##-- a
  $ap = pdl(double, [[0,   1/4, 1/4],
		     [1/2, 1/2, 1/4],
		     [1/4, 1/4, 1/2]]);
  $a = log(normpdl($ap));

  ##-- b
  $bp = pdl(double, [[1,   0,   0],
		     [0, 3/4, 1/4],
		     [0, 1/4, 3/4]]);
  $b = log(normpdl($bp));

  ##-- pi
  $pip = pdl(double,[1,0,0]); ## initial probability: bos
  $pi = log($pip/$pip->sum);

  ##-- o
  $o = pdl([0,1,2,0]); $t=$o->nelem;
}

use vars qw($delta $deltap $psi $t $path0);
sub vtest2 {
  #hmmviterbi($a,$b,$pi, $o, ($delta=zeroes(double,$n,$t)), ($psi=zeroes(long,$n,$t)));
  ($delta,$psi) = hmmviterbi($a,$b,$pi, $o);
  $deltap = $delta->exp;

  $path = hmmpath($psi, sequence($n));
  $path0 = $path->slice(",(0)");
}

##---------------------------------------------------------------------
## DUMMY
##---------------------------------------------------------------------
foreach $i (0..100) {
  print "--dummy($i)--\n";
}

