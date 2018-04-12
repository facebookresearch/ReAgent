/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* List of projects/orgs using your project for the users page */
const users = [
    /*
    {
        caption: 'User1',
        image: '/test-site/img/docusaurus.svg',
        infoLink: 'https://www.facebook.com',
        pinned: true,
    },
    */
];

const siteConfig = {
    title: 'BlueWhale',
    tagline: 'Large-Scale Reinforcement Learning Engine',
    url: 'https://facebookresearch.github.io',
    baseUrl: '/BlueWhale/',
    headerLinks: [
        {href: 'https://github.com/facebookresearch/BlueWhale', label:'Fork us on GitHub!'},
    ],
    users,
    /* path to images for header/footer */
    headerIcon: 'img/logo/icon_light.png',
    footerIcon: 'img/logo/icon_no_background.png',
    favicon: 'img/logo/icon_light.png',
    /* colors for website */
    colors: {
        primaryColor: '#28ACE2',
        secondaryColor: '#1E81AA',
    },
    /* custom fonts for website */
    /*fonts: {
       myFont: [
       "Times New Roman",
       "Serif"
       ],
       myOtherFont: [
       "-apple-system",
       "system-ui"
       ]
       },*/
    // This copyright info is used in /core/Footer.js and blog rss/atom feeds.
    copyright:
             'Copyright © ' +
             new Date().getFullYear() +
             ' Facebook',
    organizationName: 'facebookresearch',
    projectName: 'BlueWhale',
    highlight: {
        // Highlight.js theme to use for syntax highlighting in code blocks
        theme: 'default',
    },
    scripts: ['https://buttons.github.io/buttons.js'],
    // You may provide arbitrary config keys to be used as needed by your template.
    repoUrl: 'https://github.com/facebookresearch/BlueWhale/site',
    /* On page navigation for the current documentation page */
    onPageNav: 'separate',
};

module.exports = siteConfig;
