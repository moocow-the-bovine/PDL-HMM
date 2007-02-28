# -*- Mode: Perl -*-
# t/03_baum.t: test Baum-Welch re-estimation

$TEST_DIR = './t';
#use lib qw(../blib/lib ../blib/arch); $TEST_DIR = '.'; # for debugging

# load common subs
use Test;
do "$TEST_DIR/common.plt";
use PDL;
use PDL::HMM;

BEGIN { plan tests=>11, todo=>[]; }

##-- test model 1:
sub testmodel1 {
  $pi = pdl(double, [.5,.5])->log;

  $a = pdl(double,    [[.4,.4],
		       [.4,.4]])->log;
  $omega = pdl(double, [.2,.2])->log;

  $b = pdl(double, [[1,0],[0,1]])->log;
}

sub emreset {
  ##-- re-est: base
  ($ea,$eb,$epi,$eomega) = hmmexpect0($a,$b,$pi,$omega);
  $etp                   = logzero;
}

sub emE {
  foreach $o (@os) {
    ##-- alpha, beta
    $fw = hmmfw($a,$b,$pi,$o);
    $bw = hmmbw($a,$b,$omega,$o);

    ##-- text-prob
    $etp->inplace->logadd(logsumover($fw->slice(",-1") + $omega));

    ##-- re-est: expect
    hmmexpect($a,$b,$pi,$omega, $o, $fw,$bw, $ea,$eb,$epi,$eomega);
  }
  $etp -= log(scalar(@os));
}

sub emM {
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

  ##-- now can compare text-probs: $etphat -- $etp
  $etpdiff = $etphat->logdiff($etp);
}

sub wantmodel1 {
  ##-- inputs
  @os = ((map { pdl([0,1]) } (1..4)),
	 (map { pdl([1,0]) } (1..2)));

  ##-- model 2: want: text-prob
  $etp_want = pdl(double, log(1/25));

  ##-- test: want: a
  $ea_want = pdl(double, [[0,2], [4,0]])->log;
  $eb_want = pdl(double, [[6,0], [0,6]])->log;
  $epi_want = pdl(double, [4,2])->log;
  $eomega_want = pdl(double, [2,4])->log;

  ##-- model 2: want: etphat
  $etphat_want  = pdl(double, log(.209876));
  $etpdiff_want = pdl(double, log(exp($etp_want)-exp($etphat_want)));

  ##-- model 2: want: maximized
  $ahat_want  = pdl(double,   [[0,  1],
			       [2/3,1/3]])->log;
  $omegahat_want = pdl(double, [1/3,2/3])->log;

  $bhat_want  = pdl(double, [[1,0],[0,1]])->log;

  $pihat_want = pdl(double, [2/3,1/3])->log;

}

##-- tests: model 1

##-- 1--5: expect
testmodel1(); wantmodel1(); emreset(); emE(); emM();
isok("E(p(O))",     all($etp->approx($etp_want)));
isok("E(f(i-->j))", all($ea->approx($ea_want)));
isok("E(f(k @ j))", all($eb->approx($eb_want)));
isok("E(f(i | BOS))", all($epi->approx($epi_want)));
isok("E(f(EOS | i))", all($eomega->approx($eomega_want)));

##-- 6--11: maximize
isok("E(^p(O))",   all($etphat_want->approx($etphat)));
isok("E(^p)-E(p)", all($etpdiff_want->approx($etpdiff)));
isok("Ahat",       all($ahat->approx($ahat_want)));
isok("Bhat",       all($bhat->approx($bhat_want)));
isok("pihat",      all($pihat->approx($pihat_want)));
isok("omegahat",   all($omegahat->approx($omegahat_want)));


print "\n";
# end of t/XX_yyyy.t

