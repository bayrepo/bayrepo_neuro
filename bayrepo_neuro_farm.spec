%global p_version 0.1
%global p_release 2
%define pkgname bayrepo-neuro-farm

%define debug_package %{nil}

Name:           %{pkgname}
Version:        %{p_version}
Release:        %{p_release}
Summary:        The Tool for easy creating of neuro net. For testing, education for experiments


License: MIT
Group: Development/Languages
Source0: %{pkgname}-%{version}.tar.gz
URL: https://security.web.cern.ch/security/recommendations/en/codetools/rats.shtml
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

BuildRequires: make gcc gsl-devel libpng-devel libcurl-devel openssl-devel cmake
Requires: gsl libpng libcurl openssl

%description
The Tool for easy creating of neuro net. For testing, education for experiments

%prep
%setup -q

%build
cmake .
make

%install
rm -rf $RPM_BUILD_ROOT
mkdir -p $RPM_BUILD_ROOT%{_sbindir}
mkdir -p $RPM_BUILD_ROOT%{_libdir}
mkdir -p $RPM_BUILD_ROOT%{_includedir}/%{pkgname}
mkdir -p $RPM_BUILD_ROOT%{_datadir}/doc/%{pkgname}

install -D -p -m 755 bayrepo_neuro_farm %{buildroot}%{_sbindir}/bayrepo_neuro_farm
install -D -p -m 644 libbayrepo_nfl.a %{buildroot}%{_libdir}/libbayrepo_nfl.a
install -D -p -m 644 neuro_web_client.h %{buildroot}%{_includedir}/%{pkgname}/neuro_web_client.h
install -D -p -m 644 civetweb/LICENSE.md %{buildroot}%{_datadir}/doc/%{pkgname}/
install -D -p -m 644 LICENSE %{buildroot}%{_datadir}/doc/%{pkgname}/

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root,-)
%doc %{_datadir}/doc/%{pkgname}/LICENSE
%doc %{_datadir}/doc/%{pkgname}/LICENSE.md
%{_sbindir}/bayrepo_neuro_farm
%{_libdir}/libbayrepo_nfl.a
%{_includedir}/%{pkgname}/neuro_web_client.h

%changelog
* Fri Nov 08 2019 Alexey Berezhok <bayrepo.info@gmail.com> 0.1-2
- Fixed restore matrixes from file

* Wed Nov 06 2019 Alexey Berezhok <bayrepo.info@gmail.com> 0.1-1
- Initial build
