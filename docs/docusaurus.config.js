// @ts-check

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "PyIndicators",
  tagline:
    "A powerful Python library for financial technical analysis indicators",
  favicon: "img/favicon.ico",

  url: "https://codingkitties.github.io",
  baseUrl: "/PyIndicators/",

  organizationName: "CodingKitties",
  projectName: "PyIndicators",
  deploymentBranch: "gh-pages",
  trailingSlash: false,

  onBrokenLinks: "throw",

  markdown: {
    format: "detect",
    hooks: {
      onBrokenMarkdownImages: "warn",
      onBrokenMarkdownLinks: "warn",
    },
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: "content",
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl:
            "https://github.com/CodingKitties/PyIndicators/tree/main/docs/",
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "PyIndicators",
        items: [
          {
            type: "docSidebar",
            sidebarId: "docs",
            position: "left",
            label: "Documentation",
          },
          {
            href: "https://github.com/CodingKitties/PyIndicators",
            label: "GitHub",
            position: "right",
          },
          {
            href: "https://pypi.org/project/pyindicators/",
            label: "PyPI",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Docs",
            items: [
              { label: "Getting Started", to: "/" },
              { label: "Trend Indicators", to: "/indicators/trend/overview" },
              {
                label: "Support & Resistance",
                to: "/indicators/support-resistance/overview",
              },
            ],
          },
          {
            title: "Community",
            items: [
              {
                label: "GitHub",
                href: "https://github.com/CodingKitties/PyIndicators",
              },
              {
                label: "PyPI",
                href: "https://pypi.org/project/pyindicators/",
              },
            ],
          },
          {
            title: "Sponsors",
            items: [
              {
                label: "Finterion",
                href: "https://www.finterion.com/",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} CodingKitties. Built with Docusaurus.`,
      },
      prism: {
        theme: require("prism-react-renderer").themes.github,
        darkTheme: require("prism-react-renderer").themes.dracula,
        additionalLanguages: ["python", "bash"],
      },
    }),
};

module.exports = config;
