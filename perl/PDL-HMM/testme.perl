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
## Forward, backward: constrained
sub tptest1q {
  model1();

  ##-- o
  $o = pdl([0,1,1]);

  ##-- oq
  $oq = zeroes(long, $k, $o->nelem)-1;
  $b->dice_axis(1,$o)->maximum_n_ind($oq);

  $oq1 = $oq->slice("0");

  $oq2 = pdl($oq);
  $oq2->slice("1,1:2") .= -1;

  ##-- alpha
  $fw = hmmfw($a,$b,$pi, $o);
  $fwp = exp($fw);

  $fwq  = hmmfwq($a,$b,$pi, $o,$oq);
  $fwqp = exp($fwq);

  $fwq1  = hmmfwq($a,$b,$pi, $o,$oq1);
  $fwq1p = exp($fwq1);

  $fwq2  = hmmfwq($a,$b,$pi, $o,$oq2);
  $fwq2p = exp($fwq2);

  $fwp_expect  = pdl(double, [[1/2,0],[0,1/5],[0,2/25]]);
  print
    ("fw:   ", (all($fwp->approx($fwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n",
     "fwq:  ", (all($fwqp->approx($fwp_expect->indexND($oq->cat($oq->yvals)->reorder(2,0,1))))
	       ? "OK" : "NOT OK."), "\n",
     "fwq1: ", (all($fwq1p->approx($fwp_expect->indexND($oq1->cat($oq1->yvals)->reorder(2,0,1))))
		? "OK" : "NOT OK."), "\n",
     "fwq2: ", ((not
		 grep {
		   $t     = $_;
		   $oq2i  = $oq2->slice(":,($t)");
		   $oq2iw = $oq2i->where($oq2i>=0);
		   !all($fwp_expect->slice(":,($t)")->index($oq2iw)
			->approx($fwq2p->slice("0:".($oq2iw->nelem-1).",($t)"), 1e-6)
		       )
		 } (0..($o->nelem-1))
		) ? "OK" : "NOT OK."), "\n",
    );


  ##-- beta
  $bw = hmmbw($a,$b,$omega, $o);
  $bwp = exp($bw);

  $bwq  = hmmbwq($a,$b,$omega, $o,$oq);
  $bwqp = exp($bwq);

  $bwq1  = hmmbwq($a,$b,$omega, $o,$oq1);
  $bwq1p = exp($bwq1);

  $bwq2  = hmmbwq($a,$b,$omega, $o,$oq2);
  $bwq2p = exp($bwq2);

  $bwp_expect  = pdl(double, [[4/125,4/125],[2/25,2/25],[1/5,1/5]]);

  print
    ("bw:   ", (all($bwp->approx($bwp_expect,1e-6))   ? "OK" : "NOT OK."), "\n",
     "bwq:  ", (all($bwqp->approx($bwp_expect->indexND($oq->cat($oq->yvals)->reorder(2,0,1))))
	       ? "OK" : "NOT OK."), "\n",
     "bwq1: ", (all($bwq1p->approx($bwp_expect->indexND($oq1->cat($oq1->yvals)->reorder(2,0,1))))
		? "OK" : "NOT OK."), "\n",
     "bwq2: ", ((not
		 grep {
		   $t     = $_;
		   $oq2i  = $oq2->slice(":,($t)");
		   $oq2iw = $oq2i->where($oq2i>=0);
		   !all($bwp_expect->slice(":,($t)")->index($oq2iw)
			->approx($bwq2p->slice("0:".($oq2iw->nelem-1).",($t)"), 1e-6)
		       )
		 } (0..($o->nelem-1))
		) ? "OK" : "NOT OK."), "\n",
    );

  ##-- text prob
  $fwtp   = sumover($fwp->slice(",-1") * $omegap);
  $fwqtp  = sumover($fwqp->slice(",-1") * $omegap->index($oq->slice(",(-1)")));
  $fwq1tp = sumover($fwq1p->slice(",-1") * $omegap->index($oq1->slice(",(-1)")));
  $oq2Ti  = $oq2->slice(",(-1)")->where($oq2->slice(",(-1)")>=0);
  $fwq2tp = sumover($fwq2p->slice("0:".($oq2Ti->nelem-1).",-1") * $omegap->index($oq2Ti));

  $o0     = $o->at(0);
  $bwtp   = sumover($bwp->slice(",0")  * $pip * $bp->slice(",($o0)"));
  $bwqtp  = sumover($bwqp->slice(",0")
		    * $pip->index($oq->slice(",(0)"))
		    * $bp->slice(",($o0)")->index($oq->slice(",(0)")));
  $bwq1tp = sumover($bwq1p->slice(",0")
		    * $pip->index($oq1->slice(",(0)"))
		    * $bp->slice(",($o0)")->index($oq1->slice(",(0)")));

  $oq20i  = $oq2->slice(":,(0)")->where($oq2->slice(":,(0)")>=0);
  $bwq2tp = sumover($bwq2p->slice("0:".($oq20i->nelem-1).",0")
		    * $pip->index($oq20i)
		    * $bp->slice(",($o0)")->index($oq20i));
  print
    ("fw==bw       ? ", (all($fwtp->approx($bwtp),1e-6) ? "OK" : "NOT OK"), "\n",
     "fwqtp==fwtp  ? ", (all($fwtp->approx($fwqtp),1e-6) ? "OK" : "NOT OK"), "\n",
     "fwq1tp==fwtp ? ", (all($fwtp->approx($fwq1tp),1e-6) ? "OK" : "NOT OK"), "\n",
     "fwq2tp==fwtp ? ", (all($fwtp->approx($fwq2tp),1e-6) ? "OK" : "NOT OK"), "\n",
     "bwqtp==bwtp  ? ", (all($bwtp->approx($bwqtp),1e-6) ? "OK" : "NOT OK"), "\n",
     "bwq1tp==bwtp ? ", (all($bwtp->approx($bwq1tp),1e-6) ? "OK" : "NOT OK"), "\n",
     "bwq2tp==bwtp ? ", (all($bwtp->approx($bwq2tp),1e-6) ? "OK" : "NOT OK"), "\n",
    );

}


##---------------------------------------------------------------------
## Re-estimation
sub emtest1i {
  model1();

  ##-- os
  @os  = ((map { pdl([0,1,0]) } (1..4)),  (map { pdl([1,0,1]) } (1..2)));

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

  $eaf = $ea->exp;
  $ebf = $eb->exp;
  $epif = $epi->exp;
  $eomegaf = $eomega->exp;
}

##-- constrained em
sub emtest1iq {
  model1();

  ##-- os
  @os  = ((map { pdl([0,1,0]) } (1..4)),  (map { pdl([1,0,1]) } (1..2))) if (!@os);
  @oqs = (map { my $tmp=zeroes(long,2,$_->nelem)-1; $b->dice_axis(1,$_)->maximum_n_ind($tmp); $tmp } @os);
  #@oqs = (map { $b->dice_axis(1,$_)->maximum_ind->slice("*1,:") } @os);
  #@oqs = (map { sequence(long,$n)->slice(":,*".($_->nelem)) } @os);

  emtest1e0q();
}
sub emtest1e0q {
  ##-- re-est: base
  ($eaq,$ebq,$epiq,$eomegaq) = hmmexpect0($a,$b,$pi,$omega);
  $etpq = logzero;
}
use vars qw($eaqf $ebqf $epiqf $eomegaqf);
sub emtest1eq {
  foreach $i (0..$#os) {
    $o  = $os[$i];
    $oq = $oqs[$i];

    ##-- alpha
    $fwq = hmmfwq($a,$b,$pi, $o,$oq); $fwqp = exp($fwq);

    ##-- beta
    $bwq  = hmmbwq($a,$b,$omega, $o,$oq); $bwqp = exp($bwq);

    ##-- text-prob
    $etpq->inplace->logadd(logsumover($fw->slice(",-1") + $omega->index($oq->slice(",(-1)"))));

    ##-- re-est: expect
    hmmexpectq($a,$b,$pi,$omega, $o,$oq, $fwq,$bwq, $eaq,$ebq,$epiq,$eomegaq);
  }
  $etpq -= log(scalar(@os));

  $eaqf = $eaq->exp;
  $ebqf = $ebq->exp;
  $epiqf = $epiq->exp;
  $eomegaqf = $eomegaq->exp;
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

sub vtest1q {
  $n = 2; $k = 3;

  $pi = pdl(double,[.6,.4])->log;
  $a = pdl(double,   [[.5, .2],
		      [.3, .5]])->log;
  $omega = pdl(double,[.2, .3])->log;
  $b = pdl(double, [[.8, .2], [.2, .8], [.4,.6]])->log;

  ##-- aliases
  ($ap,$bp,$pip,$omegap) = map { exp($_) } ($a,$b,$pi,$omega);

  ##-- o
  $o     = pdl([0,1,2]); $t=$o->nelem;
  $blogf = $a->logsumover->logadd($pi) + $b;
  $oq    = zeroes(long,$n,$t)-1;
  $blogf->dice_axis(1,$o)->maximum_n_ind($oq);

  ##-- delta, psi
  ($deltaq,$psiq) = hmmviterbiq($a,$b,$pi, $o,$oq);
  our $deltaqp = $deltaq->exp;

  our $pathq  = hmmpathq($oq,$psiq, sequence($oq->dim(0)));
  our $qbestq = maximum_ind($deltaq->slice(",-1") + $omega->index($oq->slice(",(-1)")))->at(0);
  our $pbestq = $pathq->slice(",($qbestq)");

  ##-- alt: native
  #$pathq  = hmmpath($psiq, sequence($oq->dim(0)));
  #$qbestq = maximum_ind($deltaq->slice(",-1") + $omega->index($oq->slice(",(-1)")))->at(0);
  #$pbestq_native = $oq->indexND($pathq->slice(",($qbestq)")->cat(sequence(long,$t))->xchg(0,1));
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

